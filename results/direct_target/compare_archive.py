"""All-candidate retrospective direct-target comparison on fixed raw draws."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits, threadpool_info

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'forecast_confirmation'
sys.path.insert(0, str(HERE))
from direct_target import full_u, block_empirical_bernstein, _kernel_bounds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def by_adjust(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    ranks = np.arange(1, len(p)+1)
    harmonic = np.sum(1/ranks)
    adjusted_sorted = np.minimum.accumulate((p[order]*len(p)*harmonic/ranks)[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(1, adjusted_sorted)
    return result


def load_inputs(task):
    archive = np.load(SOURCE/task/'forecast_archive.npz', allow_pickle=False)
    indices = np.load(SOURCE/task/'sampling_indices.npz', allow_pickle=False)
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    ti = ['appliances','metro'].index(task)
    selection = archive['selection']
    n_population = int(selection.sum())
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(protocol['seed']+1000+ti).spawn(3)]
    regenerated = [streams[0].integers(n_population,size=4096),
                   streams[1].integers(n_population,size=(8192,2)),
                   streams[2].integers(n_population,size=12288)]
    for name, expected in zip(['training','validation','evaluation'],regenerated):
        assert np.array_equal(indices[name],expected), name
    pooled = np.concatenate([indices['training'],indices['validation'].ravel(),indices['evaluation']])
    assert len(pooled) == 32768 and np.array_equal(pooled, indices['all_rows'])
    return archive, selection, pooled


def case_result(case):
    task, baseline, candidate = case
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        archive, selection, pooled = load_inputs(task)
        c = archive[baseline+'__category'][selection].astype(int)
        v = archive[baseline+'__'+candidate][selection]
        w = archive['y'][selection]
        population_size = len(c)
        counts = np.bincount(c, minlength=32)
        p = {j: float(count)/population_size for j,count in enumerate(counts) if count}
        assert len(p) >= 1 and set(np.unique(c)) == set(p)
        # This census is a diagnostic only; neither sampling estimator uses it.
        rv = (rankdata(v,method='average')-0.5)/population_size
        rw = (rankdata(w,method='average')-0.5)/population_size
        mv = np.divide(np.bincount(c,weights=rv,minlength=32),counts,
                       out=np.zeros(32),where=counts>0)
        mw = np.divide(np.bincount(c,weights=rw,minlength=32),counts,
                       out=np.zeros(32),where=counts>0)
        exact_target = float(np.mean((rv-mv[c])*(rw-mw[c])))
        old = pd.read_csv(SOURCE/'selection_certificates.csv')
        prior = old[(old.task==task)&(old.baseline==baseline)&(old.candidate==candidate)
                    &(old.method=='reference_u')]
        assert len(prior) == 1
        assert abs(exact_target-prior.exact_target.iloc[0]) < 2e-13
        x, y, z = v[pooled], w[pooled], c[pooled]
        lower_k, upper_k = _kernel_bounds(p)
        width = upper_k-lower_k
        q = len(pooled)//4
        family_first = .05/(8*np.sum(1/np.arange(1,9)))
        full_started = time.perf_counter()
        u = full_u(x,y,z,p)
        full_seconds = time.perf_counter()-full_started
        d = max(u['estimate'],0.)
        logp_full = -2*q*(d/width)**2
        full_p = float(np.exp(logp_full))
        full_r = width*math.sqrt(-math.log(.05)/(2*q))
        full_first_r = width*math.sqrt(-math.log(family_first)/(2*q))
        block_started = time.perf_counter()
        eb = block_empirical_bernstein(x,y,z,p,alpha=.05)
        block_seconds = time.perf_counter()-block_started
        a = math.sqrt(2*eb['sample_variance']/q)
        b = 7*width/(3*(q-1))
        d = max(eb['estimate'],0.)
        root = 2*d/(a+math.sqrt(a*a+4*b*d)) if d else 0.
        logp_block = min(0., math.log(2.)-root**2)
        block_p = float(np.exp(logp_block))
        first_log = math.log(2.)-math.log(family_first)
        eb_first_r = a*math.sqrt(first_log)+b*first_log
        common = dict(task=task,baseline=baseline,candidate=candidate,raw_draw_calls=len(pooled),
                      distinct_sampled_rows=len(np.unique(pooled)),population_rows=population_size,
                      declared_categories=32,positive_categories=len(p),p_min=min(p.values()),
                      exact_archive_target=exact_target,independent_blocks=q,kernel_lower=lower_k,
                      kernel_upper=upper_k,kernel_range=width,alpha=.05,
                      by_first_threshold=family_first,whole_case_seconds=time.perf_counter()-started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rows = [dict(common,method='direct_full_u_hoeffding',estimate=u['estimate'],
                     radius=full_r,lower=u['estimate']-full_r,p=full_p,log_p=logp_full,
                     first_threshold_lower=u['estimate']-full_first_r,sample_variance=None,
                     arithmetic_seconds=full_seconds,first_term=u['first_term'],second_term=u['second_term']),
                dict(common,method='direct_block_empirical_bernstein',estimate=eb['estimate'],
                     radius=eb['radius'],lower=eb['lower'],p=block_p,log_p=logp_block,
                     first_threshold_lower=eb['estimate']-eb_first_r,sample_variance=eb['sample_variance'],
                     arithmetic_seconds=block_seconds,first_term=None,second_term=None)]
        for row in rows:
            assert all(math.isfinite(float(row[k])) for k in ['estimate','radius','lower','p','kernel_range'])
            assert 0 <= row['p'] <= 1
            row['lower_covers_archive_target'] = bool(row['lower'] <= exact_target+1e-13)
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args = ap.parse_args()
    assert args.workers == 4
    assert not args.output.exists()
    args.output.mkdir()
    protocol = json.loads((HERE/'protocol.json').read_text())
    seal = json.loads((HERE/'input_hashes.json').read_text())
    for name,digest in seal['sha256'].items():
        assert sha(HERE/name) == digest, name
    for task in protocol['tasks']:
        # Check the design and draw identity before opening candidate outcomes.
        load_inputs(task)
    cases = [(t,b,c) for t in protocol['tasks'] for b in protocol['baselines'] for c in protocol['candidates']]
    assert len(cases) == 32
    before = {str(p.relative_to(SOURCE)):sha(p) for p in [SOURCE/'frozen_selections.csv',
              SOURCE/'confirmation_results.csv',SOURCE/'confirmation_dependent_bounds.csv']}
    execution = dict(started_utc=utc(),protocol_sha256=sha(HERE/'protocol.json'),
                     input_hashes_sha256=sha(HERE/'input_hashes.json'),workers=args.workers,
                     script_sha256=sha(__file__),
                     hardware=dict(platform=platform.platform(),python=platform.python_version(),
                                   numpy=np.__version__,logical_cpus=os.cpu_count(),
                                   threadpool_info=threadpool_info()),
                     original_confirmation_hashes=before,scope=protocol['status'])
    (args.output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(case_result,case):case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            result = future.result()
            rows.extend(result)
            (args.output/('__'.join(case)+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'completed':case,'completed_cases':len(rows)//2,
                              'arithmetic_seconds':{r['method']:r['arithmetic_seconds'] for r in result}}),flush=True)
    frame = pd.DataFrame(rows).sort_values(['task','baseline','candidate','method'])
    frame['p_BY'] = frame.groupby(['task','baseline','method'])['p'].transform(by_adjust)
    frame['retained'] = frame.p_BY <= .05
    assert len(frame) == 64
    assert frame.groupby(['task','baseline','method']).size().eq(8).all()
    frame.to_csv(args.output/'all_candidates.csv',index=False)
    summary = frame.groupby('method').agg(candidates=('retained','size'),retained=('retained','sum'),
                  lower_cover_count=('lower_covers_archive_target','sum'),
                  minimum_p=('p','min'),minimum_p_BY=('p_BY','min'),
                  median_arithmetic_seconds=('arithmetic_seconds','median'))
    summary.to_csv(args.output/'summary.csv')
    assert all(sha(SOURCE/name)==digest for name,digest in before.items())
    receipt = dict(completed_utc=utc(),rows=len(frame),candidates=32,methods=2,
                   by_families_per_method=4,by_family_size=8,
                   all_original_confirmation_files_unchanged=True,
                   original_primary_cross_task_confirmation_success='0/2, unchanged',
                   new_confirmation_or_reselection=False,
                   results_sha256=sha(args.output/'all_candidates.csv'),
                   summary=summary.reset_index().to_dict(orient='records'),
                   interpretation='Retrospective valid but deliberately conservative raw-target inference; not a strongest-calibration or power-superiority comparison')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
