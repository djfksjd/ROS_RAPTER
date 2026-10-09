"""Launch the optional Codex profile with the local UNI_AI key, without shell export.
No network call, key printing, shell interpretation or global default changes.
"""
import os, re, shutil, sys
from pathlib import Path

def main():
    path=Path(__file__).resolve().parents[1]/'.env'
    if not path.is_file():raise SystemExit('Local .env is missing.')
    values=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        match=re.match(r'^\s*(?:export\s+)?UNI_AI\s*=\s*(.*?)\s*$',line)
        if match:
            value=match.group(1)
            if len(value)>=2 and value[0]==value[-1] and value[0] in '\"\'':value=value[1:-1]
            values.append(value)
    if len(values)!=1 or not values[0] or values[0]=='YOUR_GATEWAY_API_KEY':raise SystemExit('Configure UNI_AI exactly once in local .env.')
    executable=shutil.which('codex')
    if not executable:raise SystemExit('Codex CLI is not installed.')
    env=dict(os.environ);env['UNI_AI']=values[0]
    os.execve(executable,[executable,'--profile','uni_ai',*sys.argv[1:]],env)
if __name__=='__main__':main()
