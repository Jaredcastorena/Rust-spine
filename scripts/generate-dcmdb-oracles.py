"""Generate synthetic DCMDb fixtures from the supplied Python package directory.

Only source code is loaded. No existing database, model, or personal memory is read.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from dcmdb import HyperParams, MemoryDB

    observations = [
        [1., 0., 0.], [0.72, 0.69, 0.], [0., 0.8, 0.6],
        [0.98, 0.12, 0.03], [0.6, 0.75, 0.1], [-0.5, 0.6, 0.5],
        [0.8, 0.4, 0.15], [0.2, 0.9, 0.3], [0.6, 0.2, 0.77],
        [0.95, 0.2, 0.1], [0.4, 0.8, 0.3], [-0.1, 0.85, 0.5],
    ]
    cases = []
    for index, (mode, variant, significance) in enumerate(itertools.product(
        ["coupled", "decoupled", "hybrid"], ["ppmi", "npmi", "shifted"], ["none", "g2", "chi2"]
    )):
        cap = 0 if index % 2 == 0 else None
        policy = dict(
            maximum_secondary_updates=cap, source_prior_weight=0.08, source_score_weight=0.13,
            pmi_mode=mode.title(), pmi_variant={"ppmi": "Positive", "npmi": "Normalized", "shifted": "Shifted"}[variant],
            pmi_shift=0.2, pmi_significance={"none": "None", "g2": "G2", "chi2": "ChiSquared"}[significance],
            significance_power=0.5, adaptive_merge_angle=False, merge_angle_margin=0., use_decayed_effective_samples=False,
        )
        hp = HyperParams.for_dense_embed(
            d=3, theta_sim=0.8, tension_promote_count=99,
            max_update_neighbors=cap, lambda_src_prior=0.08, lambda_src_score=0.13,
            pmi_mode=mode, pmi_variant=variant, pmi_shift=0.2, pmi_significance=significance,
            lambda_graph=0.2, lambda_sem=0.7, lambda_mmr=0.25,
        )
        memory = MemoryDB(hp)
        for i, value in enumerate(observations):
            # Python's cached ANN index is not invalidated after centroid updates.
            # Rust queries current centroids exactly. Refresh the reference index
            # so this fixture compares the equations without preserving that bug.
            memory._rebuild_index(force_brute=True)
            memory.update(np.array(value), t=float(i+1), src="archive" if i % 3 == 0 else "live")
        nodes = []
        for nid, node in memory.nodes.items():
            nodes.append(dict(id=nid, centroid=node.mu.tolist(), sufficient_sum=node.S.tolist(),
                              effective_count=node.N, weight=node.w, kappa=node.kappa, confidence=node.c,
                              tau=node.tau, visits=node.visits, source_counts=node.source_counts,
                              tensioned=node.tension is not None))
        queries = []
        for value, source in [([0.9, 0.2, 0.1], None), ([0.2, 0.8, 0.5], "archive"), ([0.6, 0.5, 0.2], "live")]:
            queries.append(dict(vector=value, source=source, hits=memory.query(np.array(value), now=15., top_k=3, src=source)))
        stats = memory.stats()
        angles = memory.merge_angle_diagnostics()
        memory.calibrate_weights_equalize(src="archive")
        calibrated = [hp.lambda_sem, hp.lambda_graph, hp.lambda_time, hp.lambda_conf, hp.lambda_src_score]
        cases.append(dict(policy=policy, nodes=nodes, graph=memory.A, queries=queries, stats=stats, angles=angles, calibrated=calibrated))
    promotions = []
    for bayesian in [False, True]:
        hp = HyperParams.for_dense_embed(d=3)
        if bayesian:
            hp.tension_promote_count, hp.min_tension_evidence = 99, 2
            hp.theta_split, hp.theta_clear = 0., 0.
        memory = MemoryDB(hp)
        values = [[1., 0., 0.], [0.96, 0.28, 0.], [0.94, 0.34, 0.02]][:3 if bayesian else 2]
        for i, value in enumerate(values):
            memory._rebuild_index(force_brute=True)
            winner = memory.update(np.array(value), t=float(i+1), src="synthetic")
        assert winner != 0
        parent = memory.nodes[0]
        promotions.append(dict(bayesian=bayesian, observations=values,
            parent_centroid=parent.mu.tolist(), parent_kappa=parent.kappa,
            parent_confidence=parent.c, parent_count=parent.N,
            child_centroid=memory.nodes[winner].mu.tolist()))
    sample = np.array(observations[:6]); sample /= np.linalg.norm(sample, axis=1, keepdims=True)
    tuned = {}
    for mode in ["archival", "summary"]:
        hp = HyperParams.auto_from_sample(sample, mode=mode)
        tuned[mode] = dict(theta=hp.theta_sim, angle=hp.phi_merge)
    merges = []
    merge_inputs = []
    for angle in [0., 0.2, 0.75, 1.4]:
        merge_inputs.extend([[float(np.cos(angle)),float(np.sin(angle)),0.]]*6)
    merge_inputs.append([-1.,0.,0.])
    for adaptive, margin, decayed in itertools.product([False,True],[-0.4,0.], [False,True]):
        hp = HyperParams.for_dense_embed(d=3, theta_sim=0.995, tension_promote_count=99,
            adaptive_phi_merge=adaptive, phi_merge_margin_deg=float(np.degrees(margin)),
            use_decayed_neff=decayed, max_merges_per_pass=1)
        db = MemoryDB(hp)
        for i,value in enumerate(merge_inputs):
            db._rebuild_index(force_brute=True)
            db.update(np.array(value), t=500. if i==len(merge_inputs)-1 else float(i+1), src="synthetic")
        count = db.consolidate_pass()
        merges.append(dict(adaptive=adaptive, margin=margin, decayed=decayed, count=count,
            nodes=[dict(centroid=n.mu.tolist(), count=n.N, kappa=n.kappa, confidence=n.c, level=n.level, children=len(n.children)) for n in db.nodes.values()]))
    result = dict(schema=1, oracle_adjustments=["Refresh exact neighbor index before each update; Python's default cache retains outdated centroids"], source_sha256=hashlib.sha256((args.source_root / "dcmdb/core.py").read_bytes()).hexdigest(), observations=observations, cases=cases, promotions=promotions, sample=sample.tolist(), tuned=tuned, merge_inputs=merge_inputs, merges=merges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Generated {len(cases)} graph policies and {len(cases)*3} recall comparisons")


if __name__ == "__main__":
    main()
