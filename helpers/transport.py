"""Bounded, cancellable transport for the original release bytes only."""
from pathlib import Path
import hashlib, os, re, subprocess, threading, time, uuid
from artifacts import dump, load, sha, utc

ARCHIVE_API = 'https://zenodo.org/api/records/7079380/files/soundscape_data.zip/content'
ARCHIVE_FILE = 'https://zenodo.org/records/7079380/files/soundscape_data.zip?download=1'
LEGACY_CHUNK = 8 * 1024 * 1024
SUBCHUNK = 2 * 1024 * 1024


class DownloadUnavailable(RuntimeError):
    pass


class Transport:
    def __init__(self, archive=None):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.processes = set()
        self.archive = Path(archive).resolve() if archive else None
        self.archive_identity = None

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in self.processes:
                if process.poll() is None: process.terminate()

    def verify_archive(self, expected_size, expected_md5):
        if not self.archive: return
        print('VERIFYING_ORIGINAL_LOCAL_ARCHIVE', str(self.archive), flush=True)
        if self.archive.stat().st_size != expected_size:
            raise RuntimeError('Local archive size differs from the frozen original release')
        digest = hashlib.md5()
        with self.archive.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''): digest.update(block)
        if digest.hexdigest() != expected_md5:
            raise RuntimeError('Local archive MD5 differs from the frozen original release')
        self.archive_identity = (self.archive.stat().st_size, self.archive.stat().st_mtime_ns)
        print('ORIGINAL_LOCAL_ARCHIVE_MD5_VERIFIED', expected_md5, flush=True)

    def _once(self, url, temporary, headers, byte_range):
        command = ['curl', '-sS', '--max-time', '45', '--connect-timeout', '10',
                   '--fail', '--location', '-D', str(headers)]
        if byte_range is not None: command += ['--range', '%d-%d' % byte_range[:2]]
        command += [url, '-o', str(temporary)]
        if self.cancelled.is_set(): raise DownloadUnavailable('Download cancelled')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self.lock:
            self.processes.add(process)
            if self.cancelled.is_set(): process.terminate()
        try:
            _, stderr = process.communicate()
            return process.returncode, stderr.strip()
        finally:
            with self.lock: self.processes.discard(process)

    def request(self, url, path, byte_range=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.download.' + uuid.uuid4().hex)
        headers = path.with_name(path.name + '.headers.' + uuid.uuid4().hex)
        is_archive = url in [ARCHIVE_API, ARCHIVE_FILE]
        if self.archive and is_archive and byte_range is not None:
            if self.archive_identity != (self.archive.stat().st_size, self.archive.stat().st_mtime_ns):
                raise RuntimeError('Verified local archive changed or was not verified')
            lo, hi, total = byte_range
            if total != self.archive_identity[0] or not 0 <= lo <= hi < total: raise RuntimeError('Invalid archive range')
            with self.archive.open('rb') as source, temporary.open('wb') as target:
                source.seek(lo); remaining = hi - lo + 1
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block: raise RuntimeError('Truncated local archive')
                    target.write(block); remaining -= len(block)
            os.replace(temporary, path)
            return {'url': url, 'range': list(byte_range), 'utc': utc(), 'bytes': path.stat().st_size,
                    'sha256': sha(path), 'transport': 'locally supplied original archive with full release MD5 verified'}
        endpoints = [url, ARCHIVE_FILE if url == ARCHIVE_API else ARCHIVE_API] if is_archive else [url]
        failures = []
        try:
            for attempt in range(4):
                if self.cancelled.is_set(): raise DownloadUnavailable('Download cancelled')
                endpoint = endpoints[attempt % len(endpoints)]
                code, message = self._once(endpoint, temporary, headers, byte_range)
                problem = message or ('curl exit ' + str(code))
                if code == 0:
                    if byte_range is not None:
                        text = headers.read_text()
                        status = re.findall(r'(?im)^HTTP/\S+\s+(\d+)', text)
                        ranges = re.findall(r'(?im)^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$', text)
                        valid = status and status[-1] == '206' and ranges and tuple(map(int, ranges[-1])) == tuple(byte_range)
                        if not valid or temporary.stat().st_size != byte_range[1] - byte_range[0] + 1:
                            problem = 'Incorrect or truncated HTTP range; response was not installed'
                        else: problem = None
                    else: problem = None
                    if problem is None:
                        os.replace(temporary, path)
                        return {'url': endpoint, 'range': list(byte_range) if byte_range else None,
                                'utc': utc(), 'bytes': path.stat().st_size, 'sha256': sha(path),
                                'response_headers_sha256': sha(headers), 'attempts': attempt + 1}
                failures.append({'attempt': attempt + 1, 'url': endpoint, 'error': problem})
                if self.cancelled.is_set(): raise DownloadUnavailable('Download cancelled')
                print('ZENODO_RETRY', attempt + 1, 'of 4:', problem, flush=True)
                if attempt < 3 and self.cancelled.wait(3 * 2**attempt): raise DownloadUnavailable('Download cancelled')
            dump(path.with_name(path.name + '.failure.json'), {'utc': utc(), 'range': byte_range, 'attempts': failures})
            raise DownloadUnavailable('Zenodo did not deliver the requested bytes after 4 bounded attempts. '
                'Verified files/chunks were retained. Retry the recovery cell in this same runtime when Zenodo responds, '
                'or supply the original soundscape_data.zip using --archive. Do not replace recordings for an HTTP failure.')
        finally:
            if temporary.exists(): temporary.unlink()

    @staticmethod
    def retained(part, receipt, expected):
        if not receipt.exists(): return None
        item = load(receipt)
        if not part.exists() or sha(part) != item['sha256'] or item['range'] != list(expected):
            raise RuntimeError('Retained HTTP chunk changed: ' + str(part))
        return item

    def chunked(self, url, path, byte_range):
        """Reuse legacy 8 MiB chunks; request missing bytes in smaller subchunks."""
        path = Path(path); lo, hi, total = byte_range
        directory = path.parent / 'transfer_chunks'; directory.mkdir(parents=True, exist_ok=True)
        parts, receipts = [], []
        for first in range(lo, hi + 1, LEGACY_CHUNK):
            last = min(hi, first + LEGACY_CHUNK - 1)
            part = directory / f'{first}-{last}.bin'; receipt = part.with_suffix('.json')
            item = self.retained(part, receipt, (first, last, total))
            if item is None:
                children, child_receipts = [], []
                for start in range(first, last + 1, SUBCHUNK):
                    end = min(last, start + SUBCHUNK - 1)
                    child = directory / 'subchunks' / f'{start}-{end}.bin'; record = child.with_suffix('.json')
                    saved = self.retained(child, record, (start, end, total))
                    if saved is None:
                        saved = self.request(url, child, (start, end, total)); dump(record, saved)
                        print('DOWNLOAD_CHUNK_VERIFIED', start, end, flush=True)
                    children.append(child); child_receipts.append(saved)
                self.assemble(children, part, last - first + 1)
                item = {'url': url, 'range': [first, last, total], 'sha256': sha(part), 'bytes': part.stat().st_size,
                        'transport': 'verified 2 MiB subchunks', 'chunks': child_receipts}
                dump(receipt, item)
            parts.append(part); receipts.append(item)
        self.assemble(parts, path, hi - lo + 1)
        return {'url': url, 'range': list(byte_range), 'sha256': sha(path), 'bytes': path.stat().st_size,
                'transport': 'legacy-compatible chunks with bounded serial transport', 'chunks': receipts}

    @staticmethod
    def assemble(parts, path, size):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.assembled.' + uuid.uuid4().hex)
        try:
            with temporary.open('wb') as output:
                for part in parts:
                    with part.open('rb') as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''): output.write(block)
            if temporary.stat().st_size != size: raise RuntimeError('Assembled range size mismatch')
            os.replace(temporary, path)
        finally:
            if temporary.exists(): temporary.unlink()
