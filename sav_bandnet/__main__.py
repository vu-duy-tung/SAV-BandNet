"""Release command line. Set paths before importing model modules."""
import argparse,os,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description='SAV-BandNet: spatial–audiovisual target extraction')
 sub=p.add_subparsers(dest='command',required=True)
 for name in ['train','infer']:
  q=sub.add_parser(name);q.add_argument('--data',required=True);q.add_argument('--output',required=True);q.add_argument('--bundle',default=os.environ.get('SAV_BUNDLE',str(Path(__file__).resolve().parents[1]/'assets/default')));q.add_argument('--device',default='cuda');q.add_argument('--limit',type=int,default=0,help='Smoke only: limit scenes');q.add_argument('--check',action='store_true',help='Validate files/counts without model or GPU initialization')
  if name=='train':
   q.add_argument('--localization-epochs',type=int,default=2);q.add_argument('--separator-epochs',type=int,default=1);q.add_argument('--extraction-epochs',type=int,default=4);q.add_argument('--identity-epochs',type=int,default=20);q.add_argument('--batch',type=int,default=2);q.add_argument('--workers',type=int,default=2);q.add_argument('--smoke-steps',type=int,default=0)
  else:
   q.add_argument('--split',choices=['tr','cv','tt'],default='tt');q.add_argument('--rank',type=int,default=0);q.add_argument('--world',type=int,default=1)
 a=p.parse_args();a.data=str(Path(a.data).resolve());a.output=str(Path(a.output).resolve());source=Path(a.bundle).resolve()
 from .utils import sha
 provenance=json.loads((source/'PROVENANCE.json').read_text())
 for name,digest in provenance['files'].items():
  if isinstance(digest,dict):digest=digest['sha256']
  assert sha(source/name)==digest, f'Bundle hash mismatch: {name}'
 counts={s:sum(bool(x.strip()) for x in (Path(a.data)/'manifests'/f'{s}.jsonl').read_text().splitlines()) for s in ['tr','cv','tt']}
 if a.command=='train' and not a.limit:assert counts==dict(tr=32768,cv=1000,tt=2000),counts
 if a.command=='infer':assert a.world>0 and 0<=a.rank<a.world
 if a.check:print(json.dumps(dict(bundle=str(source),counts=counts,model_initialized=False)));return
 if a.command=='train':
  target=Path(a.output)/'bundle';target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copytree(source,target)
  os.environ['SAV_BUNDLE']=str(target)
  from .train import run
 else:
  os.environ['SAV_BUNDLE']=str(source)
  from .infer import run
 run(a)
if __name__=='__main__':main()
