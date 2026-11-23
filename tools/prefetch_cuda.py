"""Resumable download of three pinned large NVIDIA wheels, verified against PyPI SHA256."""
import argparse,concurrent.futures,hashlib,json,subprocess,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parents[1]
NAMES=['nvidia-cufft-cu12','nvidia-curand-cu12','nvidia-nccl-cu12']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=R/'.wheelhouse');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    pins=dict(line.split('==',1) for line in (R/'requirements-blackwell.lock').read_text().splitlines() if '==' in line)
    if a.dry_run:
        print(json.dumps({name:pins[name] for name in NAMES},indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    def one(name):
        with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{pins[name]}/json',timeout=30) as f:metadata=json.load(f)
        wheels=[x for x in metadata['urls'] if '-py3-none-' in x['filename'] and x['filename'].endswith('x86_64.whl')]
        if len(wheels)!=1:raise RuntimeError(f'Expected one Linux x86_64 wheel for {name}, found {len(wheels)}')
        item=wheels[0];dest=a.output/item['filename'];expected=item['digests']['sha256']
        if dest.exists():
            if sha(dest)!=expected:raise RuntimeError(f'Wrong cached wheel hash: {dest.name}')
        else:
            partial=dest.with_suffix(dest.suffix+'.partial')
            subprocess.run(['curl','--fail','--show-error','--silent','--location','--connect-timeout','15','--max-time','300','--retry','5','--retry-all-errors','--continue-at','-','--output',str(partial),item['url']],check=True)
            if sha(partial)!=expected:raise RuntimeError(f'Wrong downloaded wheel hash: {dest.name}')
            partial.rename(dest)
        print(f'Verified {dest.name}',flush=True)
        return {'package':name,'version':pins[name],'filename':dest.name,'sha256':expected,'bytes':dest.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,NAMES))
    (a.output/'MANIFEST.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
