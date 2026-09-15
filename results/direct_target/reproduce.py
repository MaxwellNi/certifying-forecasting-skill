"""Verify the direct-target formulas and all archived block/BY calculations."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((HERE/'input_hashes.json').read_text())
    for name, digest in inputs['sha256'].items():
        assert sha(HERE/name) == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-B', str(HERE/'verify.py'), '--skip-timings',
                    '--output', str(args.output/'formulas.json')], check=True)
    subprocess.run([sys.executable, '-B', str(HERE/'verify_archive.py'),
                    '--output', str(args.output/'archive.json')], check=True)
    formulas = json.loads((args.output/'formulas.json').read_text())
    archive = json.loads((args.output/'archive.json').read_text())
    assert formulas['passed'] and archive['passed']
    result = {
        'status': 'PASS',
        'input_hashes_checked': len(inputs['sha256']),
        'rational_full_sum_cases': formulas['brute_force_cases'],
        'maximum_full_sum_error': formulas['max_absolute_error'],
        'archive_candidates': archive['candidate_count'],
        'archive_result_rows': archive['result_rows'],
        'raw_block_values_recomputed': archive['block_values_independently_recomputed'],
        'by_families': archive['families'],
        'by_family_size': archive['family_size'],
        'retained': archive['retentions'],
        'original_confirmation_unchanged': archive['original_confirmation_unchanged'],
        'full_archive_quadratic_estimates_recomputed': False,
        'scope': 'Exact small-sample algorithm checks and all saved raw-block, confidence-formula and family calculations; no new confirmation or superiority claim',
    }
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
