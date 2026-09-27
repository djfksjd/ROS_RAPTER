"""Download pinned public NanoJev inputs; never reads or uploads .env."""
import json
from pathlib import Path
import subprocess
from huggingface_hub import snapshot_download

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'ai/models.json').read_text())['nanojev']
vendor=root/'vendor/NanoJev'
if not vendor.exists():
    vendor.parent.mkdir(exist_ok=True)
    subprocess.run(['git','clone',manifest['repository'],str(vendor)],check=True)
status=subprocess.check_output(['git','-C',str(vendor),'status','--porcelain']).strip()
if status:raise SystemExit('Upstream checkout has local edits; refusing to overwrite them')
subprocess.run(['git','-C',str(vendor),'checkout','--detach',manifest['code_revision']],check=True)
snapshot_download(manifest['model_repository'],revision=manifest['model_revision'],
    local_dir=root/'models/NanoJev',allow_patterns=[
        'best.safetensors','config.json','tokenizer/*','backbone_config/*'])
print('Pinned NanoJev source and checkpoint ready.')
