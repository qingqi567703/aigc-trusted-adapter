"""Strip only optimizer/training state from our own checkpoints, retaining exact weights."""
import argparse,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    source=ROOT/args.source;output=ROOT/args.output
    if output.exists():raise FileExistsError(output)
    saved=torch.load(str(source),map_location='cpu',mmap=True,weights_only=False)
    value={key:saved[key] for key in ('model','config','variant','step')}
    value.update(format='inference_only_exact_fp32',source_checkpoint_sha256=sha(source))
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(value,str(output))
    reloaded=torch.load(str(output),map_location='cpu',mmap=True,weights_only=False)
    assert value['model'].keys()==reloaded['model'].keys()
    assert all(torch.equal(t,reloaded['model'][k]) for k,t in value['model'].items())
    manifest={'path':output.relative_to(ROOT).as_posix(),'bytes':output.stat().st_size,'sha256':sha(output),
              'source_checkpoint_sha256':value['source_checkpoint_sha256'],'step':value['step'],
              'variant':value['variant'],'exact_weight_equality_verified':True,'can_resume_training':False}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':main()
