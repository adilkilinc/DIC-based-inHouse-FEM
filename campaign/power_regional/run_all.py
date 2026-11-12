"""Run the four power cases sequentially after the baseline reproduction passes."""
import subprocess
import sys
import time
import psutil
import common as C

gate = C.read_json(C.HERE / 'results/dataset1_mnt_linear/run_meta.json')
assert gate['completed_ok'] and max(gate['baseline_reproduction'].values()) < 1e-5
for cid in ('dataset1_mnt', 'dataset1_mt', 'dataset2_mnt', 'dataset2_mt'):
    out = C.HERE / 'results' / (cid + '_power')
    if (out / 'run_meta.json').exists():
        assert C.read_json(out / 'run_meta.json')['completed_ok']
        print('Already completed:', cid, flush=True)
        continue
    if psutil.virtual_memory().available < 8 * 2**30:
        raise SystemExit('Less than 8 GiB available before starting ' + cid)
    print('Starting:', cid, flush=True)
    with (C.HERE / (cid + '_console.log')).open('w', encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable, '-B', str(C.HERE/'run_case.py'), '--case', cid,
                              '--assignment', 'power', '--threads', '8'], cwd=C.HERE,
                             stdout=log, stderr=subprocess.STDOUT)
        monitor = psutil.Process(p.pid)
        peak = 0
        while p.poll() is None:
            try:
                peak = max(peak, monitor.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(1)
        if p.returncode:
            raise SystemExit(f'{cid} failed with code {p.returncode}; inspect {log.name}')
    C.write_json(out / 'resource_usage.json', dict(peak_rss_bytes=peak, threads=8))
    print(f'Completed: {cid}, peak RSS {peak/2**30:.2f} GiB', flush=True)
print('All four regional power FEM cases completed.', flush=True)
