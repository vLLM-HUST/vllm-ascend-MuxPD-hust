import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

import httpx

p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--base-url',required=True);p.add_argument('--producer-sha',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--baseline-only',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
manifest_path=out/(a.stage+'-manifest.json');m=json.loads(manifest_path.read_text());pid=m['api_pid']
with httpx.Client(trust_env=False,timeout=2) as c:
    for _ in range(600):
        try:
            if c.get(a.base_url+'/health').status_code==200:break
        except httpx.HTTPError:pass
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':raise RuntimeError('API exited before health')
        time.sleep(2)
    else:raise TimeoutError('startup')
print(a.stage,'ready',flush=True)
m['ready_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['cpu_stat_before']=Path('/sys/fs/cgroup/cpu.stat').read_text();m['process_argv']=Path(f'/proc/{pid}/cmdline').read_text().split('\0')[:-1]
allow={'ASCEND_RT_VISIBLE_DEVICES','OMP_NUM_THREADS','VLLM_PLUGINS','VLLM_VERSION','VLLM_USE_V2_MODEL_RUNNER','VLLM_WORKER_MULTIPROC_METHOD','PYTHONPATH'}
env=dict(x.split('=',1) for x in Path(f'/proc/{pid}/environ').read_text().split('\0') if '=' in x)
m['environment_allowlist']={k:v for k,v in env.items() if k in allow}
m['benchmark_sha256']=a.producer_sha;manifest_path.write_text(json.dumps(m,indent=2)+'\n')
workloads=[('decode-control',['--prefill-requests','0']),('p8k',[]),('p12k',['--prefill-input-tokens','12000']),('p12k-burst',['--prefill-input-tokens','12000','--prefill-requests','3'])]
commands=[]
for name,extra in workloads:
    commands.append(['scripts/benchmark_contention.py','--base-url',a.base_url,'--prefill-start-delay','0.2','--repeats',str(a.repeats),'--label',a.stage+'-'+name,'--output',str(out/(a.stage+'-'+name+'.json')),*extra])
if a.baseline_only:commands.append(['scripts/compare_outputs.py','--baseline-url',a.base_url,'--baseline-only','--repeats','2','--output',str(out/(a.stage+'-correctness.json'))])
for args in commands:
    assert hashlib.sha256((root/'scripts/benchmark_contention.py').read_bytes()).hexdigest()==a.producer_sha,'Benchmark producer changed'
    output=Path(args[args.index('--output')+1]);log=output.with_suffix('.client.log')
    cmd=['/root/workspace/.venv/bin/python',*args]
    row={'argv':cmd,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'producer_sha256':hashlib.sha256((root/args[0]).read_bytes()).hexdigest()}
    print('RUN',output.name,flush=True)
    with log.open('w') as f:r=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
    row.update(returncode=r.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),client_log=str(log.relative_to(root)))
    if output.exists():row['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    m['client_runs'].append(row);m['cpu_stat_after']=Path('/sys/fs/cgroup/cpu.stat').read_text();manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    print('DONE',output.name,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
print(a.stage,'all benchmarks complete',flush=True)
