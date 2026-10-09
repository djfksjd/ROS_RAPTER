from pathlib import Path
import argparse,json,gzip,hashlib
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
base=Path(__file__).resolve().parent;root=base.parents[2]
entries=json.loads((base/'artifacts.json').read_text());pending=[]
for entry in entries:
 src=base/entry['artifact'];raw=src.read_bytes();raw=gzip.decompress(raw) if entry['gzip'] else raw
 assert hashlib.sha256(raw).hexdigest()==entry['sha256'],src.name
 dest=root/entry['restore_path'];assert dest.resolve().is_relative_to(root)
 if dest.exists():
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==entry['sha256'],'Existing different artifact: '+str(dest)
 else:pending.append((dest,raw))
if args.apply:
 for dest,raw in pending:
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
print('Verified',len(entries),'artifacts; missing',len(pending),'applied',bool(args.apply))
