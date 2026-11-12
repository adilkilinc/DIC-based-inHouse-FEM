"""Freeze four matched regional cases; read original results without modifying them."""
import shutil
import numpy as np
import common as C


def main():
    manifest_path = C.HERE / 'inputs/manifest.json'
    if manifest_path.exists():
        raise SystemExit('Prepared inputs already exist; refusing to overwrite')
    cfg = C.read_json(C.LOCKED / 'campaign.json')
    mc = cfg['meso']
    report = dict(study=C.HERE.name, settings={k: mc[k] for k in (
        'E_mpa', 'nu', 'n_exp', 'n_inc', 'max_nr', 'nr_tol', 'snapshot_fractions')},
        table=cfg['hardening_tables'][mc['table']], cases={}, maps={},
        scope='Four moderate-mesh regional power cases; existing linear baselines; proportional final DIC boundaries')
    solver = C.REPO / 'mesoscale/solver/fem_tri.py'
    report['solver_sha256'] = C.sha256(solver)
    (C.HERE / 'solver').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(solver, C.HERE / 'solver/fem_tri.py')
    for ds in ('dataset1', 'dataset2'):
        key = cfg['identification'][ds]['production_key']
        old = C.LOCKED / '0_identification' / ds / key / 'off_stress'
        new = C.POWER / '0_identification' / ds / key / 'kpow'
        diag = C.read_json(new / 'diagnostics.json')
        assert diag['exponent_n'] == mc['n_exp']
        assert diag['gate_max_abs_diff'] < 1e-5 and diag['gate_nan_mismatch'] == 0
        for name in ('K_coef_filled.npy', 'yield_macro_filled.npy'):
            assert C.sha256(old / name) == diag['source_sha256'][name], (ds, name)
        assert C.sha256(new / 'yield_macro_filled.npy') == C.sha256(old / 'yield_macro_filled.npy')
        scale = cfg['datasets'][ds]['K_scale_mpa']
        linear = C.fill4(np.load(old / 'K_coef_filled.npy').astype(float) * scale)
        power = C.fill4(np.load(new / 'K_coef_filled.npy').astype(float) * scale)
        assert abs(power.mean() / scale - diag['power_filled']['mean']) < 1e-6
        source_shape = tuple(cfg['datasets'][ds]['source_shape'])
        assert power.shape == linear.shape == source_shape
        off = cfg['field_geometry'][ds]['crop_offset']
        shape = cfg['field_geometry'][ds]['field_shape']
        sl = (slice(off[0], off[0]+shape[0]), slice(off[1], off[1]+shape[1]))
        linear, power = linear[sl], power[sl]
        report['maps'][ds] = dict(source=str(new), power_sha256=C.sha256(new / 'K_coef_filled.npy'),
            linear_sha256=C.sha256(old / 'K_coef_filled.npy'), diagnostics=diag,
            linear_mean_mpa=float(linear.mean()), power_mean_mpa=float(power.mean()),
            field_shape=shape, crop_offset=off, pixel_size_mm=cfg['datasets'][ds]['pixel_size_um']/1000)
        for short, mesh in C.MESHES.items():
            rd = C.baseline_dir(ds, short)
            baseline = C.read_json(rd / 'run_meta.json')
            assert baseline['completed_ok'] and baseline['solver_sha256'] == report['solver_sha256']
            if ds == 'dataset1':
                pf = C.LOCKED / 'meso/0_inputs/problems' / (mesh + '.npz')
                mf = C.LOCKED / 'meso/1_materials' / (mesh + '.npz')
                sf = C.LOCKED / 'meso/0_inputs/segmentation' / (
                    'GB_without_twins_4.npy' if short == 'mnt' else 'segmented_image.npy')
                labels = np.load(sf)
                with np.load(mf) as z:
                    sy, kold = z['sy0_dic'], z['K_dic']
                assert C.sha256(mf) == baseline['input_sha256']['materials']
            else:
                pf = C.LOCKED / 'meso_d2/1_problems' / (mesh + '.npz')
                sf = C.LOCKED / 'meso_d2/0_inputs/partitions_d2.npz'
                with np.load(sf) as z:
                    labels = z['twin_merged' if short == 'mnt' else 'twin_resolved']
                with np.load(pf) as z:
                    sy, kold = z['sy0_dic'], z['K_dic']
            assert C.sha256(pf) == baseline['input_sha256']['problem']
            with np.load(pf) as z:
                pb = {k: z[k] for k in ('coords', 'elements', 'mat_id', 'bc_nodes', 'bc_u_dic', 'segmentation_label')}
            regs = pb['segmentation_label'].astype(np.int64)
            km_lin, cnt = C.label_means(labels, linear)
            km_pow, _ = C.label_means(labels, power)
            assert (cnt[regs] > 0).all()
            err = float(np.max(np.abs(km_lin[regs] - kold)))
            assert err < 1e-7, (ds, mesh, 'linear regional reconstruction differs', err)
            weights = C.areas(pb['coords'], pb['elements'])
            weights /= weights.sum()
            mids = pb['mat_id'].astype(np.int64)
            assert abs(np.sum(weights * sy[mids]) - baseline['material_mean_yield_area_weighted']) < 1e-7
            cid = ds + '_' + short
            dest = C.HERE / 'inputs' / (cid + '.npz')
            dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest, **pb, sy0=sy, K_linear=kold, K_power=km_pow[regs])
            report['cases'][cid] = dict(dataset=ds, mesh=mesh, short=short, input_sha256=C.sha256(dest),
                source_problem=str(pf), source_problem_sha256=C.sha256(pf),
                segmentation_source=str(sf), segmentation_sha256=C.sha256(sf), baseline=str(rd),
                linear_rebuild_max_abs_mpa=err, nodes=len(pb['coords']), elements=len(pb['elements']),
                regions=len(sy), yield_mean_mpa=float(np.sum(weights*sy[mids])),
                K_linear_mean_mpa=float(np.sum(weights*kold[mids])),
                K_power_mean_mpa=float(np.sum(weights*km_pow[regs][mids])))
            print(cid, report['cases'][cid], flush=True)
    C.write_json(manifest_path, report)
    print('All four prepared cases reproduce the locked regional linear K and retain their original yield, geometry and boundaries.')


if __name__ == '__main__':
    main()
