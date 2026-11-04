"""Install the frozen CPU analysis interpreter without system venv/ensurepip."""
from pathlib import Path
import argparse, hashlib, json, os, platform, subprocess, tarfile
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','AMD64']:
        raise RuntimeError('This installer targets Colab Linux x86_64; local verification uses an existing matching interpreter.')
    directory=args.directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'dependencies.json').read_text());bootstrap=spec['bootstrap'];archive=directory/'micromamba.tar.bz2'
    if not archive.exists():
        temp=archive.with_name(archive.name+'.tmp')
        subprocess.run(['curl','-sS','--max-time','300','--fail','--location','--retry','3','--retry-all-errors',bootstrap['url'],'-o',str(temp)],check=True)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=bootstrap['sha256']:raise RuntimeError('Bootstrap download checksum mismatch')
        os.replace(temp,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=bootstrap['sha256']:raise RuntimeError('Bootstrap archive checksum mismatch')
    binary=directory/'micromamba'
    with tarfile.open(archive) as tar:
        member=tar.getmember('bin/micromamba')
        if not member.isfile():raise RuntimeError('Invalid bootstrap member')
        raw=tar.extractfile(member).read()
    if not binary.exists() or binary.read_bytes()!=raw:binary.write_bytes(raw);binary.chmod(0o755)
    prefix=directory/'analysis';python=prefix/'bin/python';env=dict(os.environ,MAMBA_ROOT_PREFIX=str(directory/'mamba_root'))
    if not python.exists():
        subprocess.run([str(binary),'create','--yes','--no-rc','--override-channels','--channel','conda-forge',
                        '--prefix',str(prefix),'python='+spec['classifier']['python'],'pip'],check=True,env=env)
    marker=directory/'installed_spec.json'
    if not marker.exists():
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check',*spec['classifier']['packages']],check=True,env=env)
    elif json.loads(marker.read_text())!=spec['classifier']:raise RuntimeError('Existing numerical environment has different settings')
    code='import platform,numpy,scipy,sklearn,threadpoolctl,json;print(json.dumps([platform.python_version(),numpy.__version__,scipy.__version__,sklearn.__version__,threadpoolctl.__version__]))'
    actual=json.loads(subprocess.run([str(python),'-c',code],check=True,capture_output=True,text=True).stdout)
    if actual!=['3.9.13','1.24.4','1.9.1','1.1.1','3.6.0']:raise RuntimeError('Installed runtime version mismatch')
    marker.write_text(json.dumps(spec['classifier'],indent=2)+'\n')
    print('CPU_ANALYSIS_RUNTIME_READY',flush=True)

if __name__=='__main__':main()
