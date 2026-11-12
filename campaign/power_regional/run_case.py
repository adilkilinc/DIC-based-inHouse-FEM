"""Run a prepared case with an unmodified, hash-checked copy of the locked solver."""
import argparse
import csv
import os
import sys
import time
import socket
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--assignment', choices=['linear', 'power'], default='power')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    os.environ.update(OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
                      OPENBLAS_NUM_THREADS='1', MKL_DYNAMIC='FALSE')
    import numpy as np
    import common as C
    m = C.read_json(C.HERE / 'inputs/manifest.json')
    case = m['cases'][a.case]
    inp = C.HERE / 'inputs' / (a.case + '.npz')
    assert C.sha256(inp) == case['input_sha256']
    assert C.sha256(C.HERE / 'solver/fem_tri.py') == m['solver_sha256']
    st, table = m['settings'], m['table']
    with np.load(inp) as z:
        pb = {k: z[k] for k in z.files}
    out = C.HERE / 'results' / (a.case + '_' + a.assignment)
    if (out / 'run_meta.json').exists():
        raise SystemExit('Completed output already exists; refusing overwrite: ' + str(out))
    print(f"Prepared {a.case} {a.assignment}: {case['elements']} elements, {case['regions']} regions", flush=True)
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    os.environ['FEM_LOG_DIR'] = str(out / 'logs')
    sys.path.insert(0, str(C.HERE / 'solver'))
    import fem_tri
    if 'pypardiso' not in fem_tri._SOLVER_NAME:
        raise SystemExit('Production requires pypardiso: ' + fem_tri._SOLVER_NAME)
    C.write_json(out / 'started.json', dict(pid=os.getpid(), case=a.case, assignment=a.assignment,
        start=time.strftime('%Y-%m-%d %H:%M:%S'), threads=a.threads, input_sha256=case['input_sha256']))
    sy, K = pb['sy0'], pb['K_' + a.assignment]
    t0 = time.time()
    r = fem_tri.run_fem_tri(pb['coords'], pb['elements'].astype(np.int64), pb['mat_id'].astype(np.int64),
        np.zeros((1, 2)), np.array([0., 1.]), pb['bc_nodes'].astype(np.int64), pb['bc_u_dic'],
        E_mod=st['E_mpa'], nu=st['nu'], N_inc=st['n_inc'], max_nr=st['max_nr'], nr_tol=st['nr_tol'],
        snapshot_fractions=st['snapshot_fractions'], verbose=True, hardening_mode='graded',
        sy0=sy, K=K, n_exp=st['n_exp'], graded=dict(ep_first=table['ep_first'], ep_max=table['ep_max'], n_pts=table['n_table']))
    elapsed = time.time() - t0
    assert all(np.isfinite(r[k]).all() for k in ('U', 'RF', 'S', 'E', 'PE', 'PEEQ'))
    berror = float(np.max(np.abs(r['U'][pb['bc_nodes']] - pb['bc_u_dic'])))
    assert berror < 1e-10
    assert float(np.max(r['PEEQ'])) < table['ep_max'] * 0.8
    for k in ('U', 'RF', 'S', 'E', 'PE', 'PEEQ'):
        np.save(out / (k + '.npy'), np.asarray(r[k], np.float32))
    fractions = sorted(r['frames'])
    np.savez_compressed(out / 'snapshots.npz', fraction=fractions,
        **{k: np.stack([r['frames'][f][k] for f in fractions]).astype(np.float32) for k in ('E', 'S', 'PEEQ')})
    h = r['history']
    with (out / 'history_scalars.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(h.keys())
        writer.writerows(zip(*h.values()))
    weights = C.areas(pb['coords'], pb['elements'])
    weights /= weights.sum()
    meta = dict(completed_ok=True, case=case, assignment=a.assignment, settings=st, table=table,
        input_sha256=case['input_sha256'], solver_sha256=m['solver_sha256'], solver=fem_tri._SOLVER_NAME,
        n_inc_accepted=int(r['n_inc']), newton_iterations=int(r['n_nr']), cutbacks=int(r['n_cutback']),
        line_searches=int(r.get('n_line_search', 0)), solver_wall_s=float(r['wall_s']), total_wall_s=elapsed,
        boundary_error_max_mm=berror, peeq_max=float(np.max(r['PEEQ'])),
        final_mean_e22=float(weights @ r['E'][:, 1]), final_mean_evm=float(weights @ C.evm(r['E'], st['nu'])),
        final_mean_s22_mpa=float(weights @ r['S'][:, 1]), final_mean_s11_mpa=float(weights @ r['S'][:, 0]),
        final_mean_svm_mpa=float(weights @ C.svm(r['S'])), rf2_top_solver=float(h['rf2_top'][-1]),
        material_mean_yield_area_weighted=float(weights @ sy[pb['mat_id']]),
        material_mean_K_area_weighted=float(weights @ K[pb['mat_id']]),
        hostname=socket.gethostname(), threads=a.threads, completed=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.assignment == 'linear':
        base = Path(case['baseline'])
        bm = C.read_json(base / 'run_meta.json')
        check = {}
        for key in ('E', 'S', 'U'):
            old = np.load(base / (key + '.npy')).astype(float)
            check[key + '_relative_l2'] = float(np.linalg.norm(r[key]-old)/np.linalg.norm(old))
        check['mean_s22_relative_error'] = abs(meta['final_mean_s22_mpa']/bm['final_mean_s22_mpa']-1)
        C.write_json(out / 'baseline_reproduction.json', check)
        assert max(check.values()) < 1e-5, check
        meta['baseline_reproduction'] = check
    C.write_json(out / 'run_meta.json', meta)
    print(f"COMPLETE {a.case} {a.assignment}: {elapsed:.1f} s; mean axial stress {meta['final_mean_s22_mpa']:.3f} MPa", flush=True)


if __name__ == '__main__':
    main()
