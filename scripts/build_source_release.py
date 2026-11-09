"""Package reviewable source and lightweight evidence, excluding credentials and large assets."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs/aigc_course_source_20261109.zip')
    args=p.parse_args()
    files=[]
    for folder in ('aigc','scripts','configs','docs'):
        files.extend(x for x in (ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc')
    files.extend(ROOT/name for name in ('README.md','requirements.txt','requirements-linux-cp310.txt','requirements-report.txt','.gitignore'))
    records=[]
    for path in sorted(files):
        relative=path.relative_to(ROOT).as_posix()
        assert path.suffix.lower() not in ('.pt','.pth','.safetensors','.pem','.key'),relative
        assert not any(part in relative.lower() for part in ('id_rsa','school_ed25519','known_hosts','gateway_cache')),relative
        if path.suffix in ('.py','.md','.json','.jsonl','.txt','.sh'):
            contents=path.read_bytes()
            for kind in (b'OPENSSH',b'RSA'):
                marker=b'-'*5+b'BEGIN '+kind+b' PRIVATE KEY'+b'-'*5
                assert marker not in contents,relative
        records.append({'path':relative,'bytes':path.stat().st_size,'sha256':digest(path)})
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in records:z.write(ROOT/item['path'],item['path'])
        z.writestr('SOURCE_MANIFEST.json',json.dumps({'files':records},indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for item in records:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    manifest={'file':output.name,'bytes':output.stat().st_size,'sha256':digest(output),'files':len(records),
              'zip_crc_and_member_sha256_verified':True,'excludes':['private credentials','model weights','COCO originals','large experiment backups']}
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':main()
