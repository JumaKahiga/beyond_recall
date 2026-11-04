"""Isolated Colab interpreters without depending on system venv/ensurepip."""
from pathlib import Path
import argparse, hashlib, json, os, platform, subprocess, tarfile
ROOT=Path(__file__).resolve().parent

def configure_cuda_paths(directory,site):
    cuda=directory/'cuda_pip';(cuda/'bin').mkdir(parents=True,exist_ok=True)
    candidates=list((site/'nvidia').glob('*/bin/ptxas'))
    bitcodes=list((site/'nvidia').glob('*/nvvm/libdevice'))
    if len(candidates)!=1 or len(bitcodes)!=1:
        raise RuntimeError('Pinned CUDA packages did not expose ptxas/libdevice; see setup log')
    for link,target in [(cuda/'bin/ptxas',candidates[0]),(cuda/'nvvm',bitcodes[0].parent)]:
        if link.is_symlink():link.unlink()
        link.symlink_to(target)
    libraries=[str(p) for p in sorted((site/'nvidia').glob('*/lib'))]
    # Keep the host driver search path after the pinned user-space CUDA libraries.
    if os.environ.get('LD_LIBRARY_PATH'):libraries.append(os.environ['LD_LIBRARY_PATH'])
    return {'PATH':str(cuda/'bin')+os.pathsep+os.environ.get('PATH',''),
            'LD_LIBRARY_PATH':os.pathsep.join(libraries),
            'XLA_FLAGS':'--xla_gpu_cuda_data_dir='+str(cuda)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--mode',choices=['statistics','analysis','full'],default='statistics')
    parser.add_argument('--embedding-device',choices=['cpu','gpu'],default='gpu');args=parser.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ['x86_64','AMD64']:
        raise RuntimeError('This bootstrap is for Linux x86_64 Colab; local verification can use declared existing interpreters.')
    directory=args.directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    authority=json.loads((ROOT/'dependencies.json').read_text())['bootstrap']
    archive=directory/'micromamba.tar.bz2'
    if not archive.exists():
        subprocess.run(['curl','-sS','--max-time','300','--fail','--location',authority['url'],'-o',str(archive)],check=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=authority['sha256']:
        raise RuntimeError('Micromamba bootstrap archive hash mismatch')
    binary=directory/'micromamba'
    if not binary.exists():
        with tarfile.open(archive) as t:
            member=t.getmember('bin/micromamba')
            if not member.isfile():raise RuntimeError('Unexpected bootstrap member')
            binary.write_bytes(t.extractfile(member).read())
        binary.chmod(0o755)
    env=dict(os.environ,MAMBA_ROOT_PREFIX=str(directory/'mamba_root'))
    def make(name,python,packages):
        prefix=directory/name;executable=prefix/'bin/python'
        specification={'python':python,'packages':packages}
        spec_path=directory/(name+'_specification.json');freeze_path=directory/(name+'_pip_freeze.txt')
        if not executable.exists():
            subprocess.run([str(binary),'create','--yes','--no-rc','--override-channels','--channel','conda-forge',
                            '--prefix',str(prefix),'python='+python,'pip'],check=True,env=env)
        version=subprocess.run([str(executable),'-c','import platform; print(platform.python_version())'],
                               check=True,text=True,capture_output=True,env=env)
        if version.stdout.strip()!=python:raise RuntimeError('Isolated Python version differs: '+name)
        if spec_path.exists():
            installed=subprocess.run([str(executable),'-m','pip','freeze'],check=True,text=True,capture_output=True,env=env)
            if json.loads(spec_path.read_text())!=specification or not freeze_path.exists() or installed.stdout!=freeze_path.read_text():
                raise RuntimeError('Installed runtime changed in an existing session: '+name)
            print('REUSING_VERIFIED_RUNTIME',name,flush=True);return str(executable)
        subprocess.run([str(executable),'-m','pip','install','--disable-pip-version-check',*packages],check=True,env=env)
        receipt=subprocess.run([str(executable),'-m','pip','freeze'],check=True,text=True,capture_output=True,env=env)
        freeze_path.write_text(receipt.stdout);spec_path.write_text(json.dumps(specification,indent=2)+'\n')
        return str(executable)
    paths={}
    paths['statistics']=make('statistics','3.11.15',['numpy==2.4.6','scipy==1.17.1'])
    if args.mode in ['analysis','full']:
        paths['classifier']=make('classifier','3.9.13',['numpy==1.24.4','scipy==1.9.1','scikit-learn==1.1.1','soundfile==0.13.1','threadpoolctl==3.6.0'])
    if args.mode=='full':
        tensorflow='tensorflow[and-cuda]==2.20.0' if args.embedding_device=='gpu' else 'tensorflow==2.20.0'
        paths['embedding']=make('embedding_'+args.embedding_device,'3.11.15',['numpy==2.4.6',tensorflow,'soundfile==0.14.0'])
        # XLA needs ptxas and libdevice as well as the CUDA runtime libraries.
        # Make pip's installed CUDA packages discoverable without altering Colab's system CUDA.
        if args.embedding_device=='gpu':
            discovery=subprocess.run([paths['embedding'],'-c','import site; print(site.getsitepackages()[0])'],
                                     check=True,text=True,capture_output=True)
            gpu_env=configure_cuda_paths(directory,Path(discovery.stdout.strip()))
            (directory/'embedding_environment.json').write_text(json.dumps(gpu_env,indent=2)+'\n')
        else:
            (directory/'embedding_environment.json').write_text('{}\n')
    (directory/'interpreters.json').write_text(json.dumps(paths,indent=2)+'\n')
    print(json.dumps(paths,indent=2))

if __name__=='__main__':main()
