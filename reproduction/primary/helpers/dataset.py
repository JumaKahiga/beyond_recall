"""Download original release bytes, with resumable entry-level ZIP extraction.

HTTP failures never imply technical recording failure. No method or target
outcomes are read here. curl validates HTTPS and every response range is checked.
"""
from artifacts import ROOT, check_recipe, dump, load, sha, utc
import argparse, hashlib, io, os, re, struct, subprocess, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

class TechnicalAudioFailure(Exception):
    @property
    def category(self):
        message=str(self)
        if message.startswith('Invalid complete-hour geometry:'):return 'invalid_audio_geometry'
        if message=='Nonfinite decoded waveform':return 'nonfinite_decoded_waveform'
        if message=='Truncated decode':return 'truncated_decode'
        return 'decoder_rejected_audio'


from transport import Transport
transport=Transport()
request=transport.request
request_chunked=transport.chunked
AUDIO_CACHE_BASE=None

class TailReader(io.RawIOBase):
    def __init__(self, tail, total):
        self.tail, self.total, self.pos = tail, total, total
        self.start = total - len(tail)
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.total + offset
        return self.pos
    def tell(self):
        return self.pos
    def seekable(self):
        return True
    def read(self, n=-1):
        if self.pos < self.start:
            raise RuntimeError('ZIP central directory exceeds retained tail')
        data = self.tail[self.pos-self.start:None if n < 0 else self.pos-self.start+n]
        self.pos += len(data)
        return data

def inventory_from_tail(tail, total):
    with zipfile.ZipFile(TailReader(tail, total)) as z:
        entries = z.infolist()
        directory_start = z.start_dir
    names, result = set(), []
    all_offsets = sorted(set([i.header_offset for i in entries] + [directory_start]))
    for info in entries:
        if info.is_dir() or info.filename.startswith('__MACOSX/') or not info.filename.lower().endswith('.flac'):
            continue
        name = Path(info.filename).name
        if not re.fullmatch(r'SSW_\d{3}_\d{8}_\d{6}Z\.flac', name) or name in names:
            raise RuntimeError('Ambiguous release recording inventory: ' + info.filename)
        names.add(name)
        next_offset = all_offsets[all_offsets.index(info.header_offset) + 1]
        result.append({'filename': name, 'archive_path': info.filename, 'bytes': info.file_size,
                       'compressed_bytes': info.compress_size, 'crc32': info.CRC,
                       'compression': info.compress_type, 'offset': info.header_offset,
                       'range_end': next_offset - 1})
    return sorted(result, key=lambda x: x['filename'])

def extract_entry(entry_file, info, destination):
    destination = Path(destination)
    tmp = destination.with_name(destination.name + '.tmp')
    crc, written = 0, 0
    with Path(entry_file).open('rb') as source:
        header = source.read(30)
        sig, version, flags, method, mt, md, header_crc, cs, us, fn, extra = struct.unpack('<IHHHHHIIIHH', header)
        if sig != 0x04034b50 or method != info['compression'] or flags & 1:
            raise RuntimeError('Invalid or encrypted ZIP local header')
        name = source.read(fn).decode('utf-8' if flags & 2048 else 'cp437')
        if name != info['archive_path']:
            raise RuntimeError('ZIP local and central filename disagreement')
        source.read(extra)
        decoder = zlib.decompressobj(-15) if method == 8 else None
        if method not in (0, 8):
            raise RuntimeError('Unsupported ZIP compression')
        remaining = info['compressed_bytes']
        with tmp.open('wb') as target:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError('Truncated ZIP entry')
                remaining -= len(chunk)
                out = decoder.decompress(chunk) if decoder else chunk
                crc = zlib.crc32(out, crc); written += len(out); target.write(out)
            if decoder:
                out = decoder.flush()
                crc = zlib.crc32(out, crc); written += len(out); target.write(out)
                if not decoder.eof:
                    raise RuntimeError('Incomplete DEFLATE stream')
    if written != info['bytes'] or crc != info['crc32']:
        raise RuntimeError('ZIP size/CRC integrity mismatch')
    os.replace(tmp, destination)

def inspect_audio(path):
    # Chunked full decode avoids holding an entire hour's multichannel waveform.
    try:
        with sf.SoundFile(path) as f:
            frames, rate, channels = len(f), f.samplerate, f.channels
            if rate != 32000 or channels < 1 or not 3600*rate <= frames < 3605*rate:
                raise TechnicalAudioFailure('Invalid complete-hour geometry: ' + repr((frames, rate, channels)))
            seen = 0
            for block in f.blocks(blocksize=160000*8, dtype='float32', always_2d=True):
                if not np.isfinite(block).all():
                    raise TechnicalAudioFailure('Nonfinite decoded waveform')
                seen += len(block)
            if seen != frames:
                raise TechnicalAudioFailure('Truncated decode')
    except (sf.LibsndfileError, RuntimeError) as error:
        raise TechnicalAudioFailure(str(error)) from error
    return {'frames': frames, 'sample_rate': rate, 'channels': channels,
            'duration_seconds': frames/rate, 'complete_windows': frames//160000,
            'discarded_tail_samples': frames % 160000, 'sha256': sha(path),
            'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__}

def prepare_entry(audio, info, url, total):
    name=info['filename']; dest=audio/name
    extraction_receipt = audio/(name+'.source.json')
    if extraction_receipt.exists():
        extracted = load(extraction_receipt)
        if not dest.exists() or sha(dest) != extracted['audio_sha256']:
            raise RuntimeError('Extracted FLAC hash mismatch: ' + name)
    else:
        compressed = audio/(name+'.zipentry')
        transfer = request_chunked(url, compressed, (info['offset'], info['range_end'], total))
        extract_entry(compressed, info, dest)
        dump(extraction_receipt, {'entry': info, 'transfer': transfer, 'audio_sha256': sha(dest),
                                  'zip_crc32_verified': True, 'archive_md5_locally_verified': False})
        # Only remove the intermediate compressed bytes that this call created.
        compressed.unlink()
    if AUDIO_CACHE_BASE is not None:
        from audio_store import save_record
        save_record(audio.parent.parent,AUDIO_CACHE_BASE,name)
    return dest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--audio-cache',type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    global AUDIO_CACHE_BASE
    AUDIO_CACHE_BASE=args.audio_cache
    recipe = check_recipe(root)
    if np.__version__ != recipe['construction']['numpy']:
        raise RuntimeError('Construction requires NumPy 1.24.4')
    source = root/'data/source'; audio = root/'data/audio'
    source.mkdir(parents=True, exist_ok=True); audio.mkdir(parents=True, exist_ok=True)
    record_path = root/'dataset.json'
    if not record_path.exists():
        request('https://zenodo.org/api/records/7079380', record_path)
    metadata = load(record_path)
    release = {item['key']: item for item in metadata['files']}
    receipts = {}
    for name, md5 in recipe['dataset']['files'].items():
        item = release[name]
        if item['checksum'] != 'md5:' + md5:
            raise RuntimeError('Release checksum differs from frozen authority: ' + name)
        if name.endswith('.zip'):
            continue
        path = source/name
        if not path.exists():
            request(item['links']['self'], path)
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        if actual != md5 or path.stat().st_size != item['size']:
            raise RuntimeError('Source file checksum/size mismatch: ' + name)
        receipts[name] = {'bytes': path.stat().st_size, 'md5_verified': actual,
                          'sha256': sha(path), 'url': item['links']['self']}
    archive = release['soundscape_data.zip']; total = archive['size']; url = archive['links']['self']
    if args.archive:
        transport.archive=args.archive.resolve();transport.verify_archive(total,recipe['dataset']['files']['soundscape_data.zip'])
    tail = source/'zip_tail.bin'
    if not tail.exists():
        request(url, tail, (total-65557, total-1, total))
    inventory = inventory_from_tail(tail.read_bytes(), total)
    permutation = np.random.Generator(np.random.PCG64(991200000)).permutation(len(inventory)).tolist()
    order = [inventory[i] for i in permutation]
    inventory_receipt = {'recipe_freeze_sha256': sha(root/'study.json'),
                         'archive': {'url': url, 'bytes': total, 'release_md5': archive['checksum'],
                                     'whole_archive_md5_locally_verified': False, 'tail_sha256': sha(tail)},
                         'inventory': inventory, 'permutation_indices': permutation,
                         'permutation_filenames': [i['filename'] for i in order], 'source_files': receipts}
    inv_path = root/'data/constructed/RECORDING_PERMUTATION.json'
    if inv_path.exists() and load(inv_path) != inventory_receipt:
        raise RuntimeError('Recording permutation or authority changed on resume')
    dump(inv_path, inventory_receipt)
    from construction import preflight
    preflight(root)
    if args.inventory_only:
        print('INVENTORY_FROZEN', len(inventory), flush=True); return
    selection_path = root/'data/constructed/RECORDING_SELECTION.json'
    selected, failures = [], []
    previous = load(selection_path) if selection_path.exists() else {'selected': [], 'technical_failures': []}
    if previous.get('status')=='RECORDINGS_COMPLETE':
        from audio_store import validate_selection
        validate_selection(previous,inventory_receipt)
        for record in previous['selected']:
            if sha(audio/record['filename'])!=record['audio']['sha256']:
                raise RuntimeError('Previously validated audio changed: '+record['filename'])
        print('REUSED_VERIFIED_RECORDING_SELECTION',len(previous['selected']),flush=True)
        return
    if AUDIO_CACHE_BASE is not None:
        from audio_store import save_source
        save_source(root,AUDIO_CACHE_BASE)
    prior = {i['filename']: i for i in previous['selected']}
    executor=ThreadPoolExecutor(max_workers=1)
    transfers={position:executor.submit(prepare_entry,audio,info,url,total) for position,info in enumerate(order[:33]) if info['filename'] not in prior}
    for position, info in enumerate(order):
        name = info['filename']; dest = audio/name
        if name in prior:
            record = prior[name]
            if sha(dest) != record['audio']['sha256']:
                raise RuntimeError('Previously validated audio changed: ' + name)
        else:
            if position not in transfers:
                transfers[position]=executor.submit(prepare_entry,audio,info,url,total)
            transfers[position].result()
            try:
                facts = inspect_audio(dest)
            except TechnicalAudioFailure as error:
                failures.append({'filename': name, 'permutation_position': position,
                                 'reason': error.category, 'sha256': sha(dest)})
                dump(selection_path, {'status': 'SELECTION_IN_PROGRESS', 'selected': selected,
                                      'technical_failures': failures, 'recipe_freeze_sha256': sha(root/'study.json')})
                print('TECHNICAL_REPLACEMENT', name, str(error), flush=True); continue
            record = {'filename': name, 'permutation_position': position, 'audio': facts}
        selected.append(record)
        complete = len(selected) == 33
        dump(selection_path, {'status': 'RECORDINGS_COMPLETE' if complete else 'SELECTION_IN_PROGRESS',
                              'selected': selected, 'technical_failures': failures,
                              'recipe_freeze_sha256': sha(root/'study.json')})
        print('RECORDING_READY', len(selected), name, flush=True)
        if complete:
            executor.shutdown(wait=True)
            return
    raise RuntimeError('CONSTRUCTION_FAILURE: fewer than 33 technically eligible recordings')

if __name__ == '__main__':
    try: main()
    finally: transport.cancel()
