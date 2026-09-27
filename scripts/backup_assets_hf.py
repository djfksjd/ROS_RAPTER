"""Back up pinned model inputs to the private HF dataset, never the workspace."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
REPO='dannykim123/ROS_RAPTER-backup'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    token=next(line.split('=',1)[1].strip().strip('\"\'') for line in
               (ROOT/'.env').read_text().splitlines() if line.strip().startswith('HF_TOKEN='))
    api=HfApi(token=token);records=[]
    def upload(local,remote):
        sha=digest(local)
        api.upload_file(path_or_fileobj=local,path_in_repo=remote,repo_id=REPO,
                        repo_type='dataset',commit_message='Back up pinned Raptor model asset')
        records.append({'path':remote,'bytes':local.stat().st_size,'sha256':sha})
        print('Uploaded',remote,flush=True)
    checkpoint=ROOT/'models/NanoJev'
    for path in sorted(checkpoint.rglob('*')):
        relative=path.relative_to(checkpoint)
        if path.is_file() and (str(relative) in ['best.safetensors','config.json']
                              or relative.parts[0] in ['tokenizer','backbone_config']):
            upload(path,'models/NanoJev/'+relative.as_posix())
    ollama=Path.home()/'.ollama/models'
    manifest=ollama/'manifests/registry.ollama.ai/library/qwen3/0.6b'
    data=json.loads(manifest.read_text())
    upload(manifest,'models/ollama/manifests/registry.ollama.ai/library/qwen3/0.6b')
    for blob in [data['config'],*data['layers']]:
        path=ollama/'blobs'/blob['digest'].replace(':','-')
        if digest(path)!=blob['digest'].split(':')[1]:raise RuntimeError('Ollama blob hash mismatch')
        upload(path,'models/ollama/blobs/'+path.name)
    with tempfile.TemporaryDirectory(prefix='raptor-source-') as tmp:
        archive=Path(tmp)/'NanoJev-source.tar.gz'
        subprocess.run(['git','-C',str(ROOT/'vendor/NanoJev'),'archive','--format=tar.gz',
                        '--output',str(archive),'76fdfc9ecdca45a9bcef17991a07d3041a87685a'],check=True)
        upload(archive,'sources/NanoJev-76fdfc9.tar.gz')
    info=api.dataset_info(REPO,files_metadata=True)
    files={f.rfilename:f for f in info.siblings}
    for record in records:
        remote=files[record['path']]
        if remote.size!=record['bytes']:raise RuntimeError('Remote size mismatch')
        if remote.lfs and remote.lfs.sha256!=record['sha256']:raise RuntimeError('Remote hash mismatch')
        record['verified']='sha256+size' if remote.lfs else 'size'
    (ROOT/'docs/evidence/model-backup-assets.json').write_text(json.dumps(
        {'repository':REPO,'hf_commit':info.sha,'files':records},indent=2))
    print('Model assets verified',flush=True)


if __name__=='__main__':main()
