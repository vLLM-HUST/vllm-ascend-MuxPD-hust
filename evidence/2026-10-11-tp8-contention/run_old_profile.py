import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

import httpx

root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
pid=int((out/'old-profile.pid').read_text())
manifest_path=out/'old-profile-manifest.json'
m=json.loads(manifest_path.read_text())
base='http://127.0.0.1:8002'
with httpx.Client(timeout=600,trust_env=False) as c:
    for _ in range(600):
        try:
            if c.get(base+'/health',timeout=2).status_code==200:break
        except httpx.HTTPError:pass
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':raise RuntimeError('API exited')
        time.sleep(2)
    else:raise TimeoutError('startup')
    print('Old profile ready; warmup',flush=True)
    warm=[]
    for length in (128,8192,12000):
        started=time.monotonic()
        r=c.post(base+'/v1/completions',json={'model':'Qwen3.5-35B-A3B','prompt':[100+i%200 for i in range(length)],'max_tokens':8,'ignore_eos':True,'seed':42,'temperature':0})
        r.raise_for_status();j=r.json();assert j['usage']['completion_tokens']==8
        warm.append({'input_tokens':length,'wall_s':time.monotonic()-started,'response':j})
    (out/'old-profile-warmup.json').write_text(json.dumps(warm,indent=2)+'\n')
    started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    r=c.post(base+'/start_profile');r.raise_for_status()
    print('Profiler started',flush=True)
    cmd=['/root/workspace/.venv/bin/python','scripts/exercise_decoupled.py','--base-url',base,'--output',str(out/'old-profile-client.json')]
    with (out/'old-profile-client.log').open('w') as f:result=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
    print('Client returned',result.returncode,flush=True)
    r=c.post(base+'/stop_profile');r.raise_for_status()
    m['profile']={'started_at':started_at,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_http_status':200,'stop_http_status':r.status_code,'argv':cmd,'returncode':result.returncode}
    m['client_sha256']=hashlib.sha256((out/'old-profile-client.json').read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    print('Old profile complete',flush=True)
    if result.returncode:raise SystemExit(result.returncode)
