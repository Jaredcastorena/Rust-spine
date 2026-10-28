"""Generate numerical fixtures from a separately supplied Python Thymos source.

Development only: requires numpy. The Rust tests consume the checked-in JSON
without Python, model downloads, credentials, or any existing memory database.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thymos-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("thymos_oracle", args.thymos_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tensor = np.array([[0.2, -0.1, 0.05], [-0.3, 0.4, 0.1], [0.2, -0.1, 0.05]])
    inputs = [[1., 2., 3.], [0.7, 0.2, -0.1], [-0.2, 0.8, 0.4], [0.3, 0.4, 0.9]]
    cases = []
    for mode in ["cosine_error", "spherical_residual", "direct_diff"]:
        for activation in ["none", "tanh", "relu", "softmax"]:
            for normalize_rows in [False, True]:
                config = module.ThymosConfig(
                    d=3, K=3, eta=0.02, decay=0.003,
                    normalize_rows=normalize_rows, valence_mode=mode,
                    activation_nonlinearity=activation, valence_temperature=0.7,
                    channel_labels=["first", "second", "third"],
                )
                thymos = module.Thymos(config, T=tensor.copy(), channel_mass=np.array([0.3, 0.5, 0.3]))
                steps = []
                for index, value in enumerate(inputs):
                    x = np.array(value)
                    expected = np.array(inputs[(index + 1) % len(inputs)])
                    eligibility = thymos.compute_valence(expected, x)
                    query = thymos.query(x).to_dict()
                    query.pop("timestamp", None)
                    raw_query = thymos.query(x, normalize_input=False).to_dict()
                    raw_query.pop("timestamp", None)
                    thymos.update(x, eligibility)
                    trajectory = thymos.step(x)
                    prediction = thymos.predict_next()
                    steps.append(dict(
                        input=value, expected=expected.tolist(), query=query,
                        unnormalized_query=raw_query, eligibility=eligibility.tolist(),
                        tensor=thymos.T.reshape(-1).tolist(), mass=thymos.channel_mass.tolist(),
                        trajectory=trajectory,
                        prediction=None if prediction is None else prediction.tolist(),
                    ))
                cases.append(dict(mode=mode, activation=activation,
                                  normalize_rows=normalize_rows, steps=steps))
    output = dict(
        schema=1,
        source_sha256=hashlib.sha256(args.thymos_source.read_bytes()).hexdigest(),
        initial_tensor=tensor.reshape(-1).tolist(), initial_mass=[0.3, 0.5, 0.3],
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Generated {len(cases)} cases / {sum(len(c['steps']) for c in cases)} observations")


if __name__ == "__main__":
    main()
