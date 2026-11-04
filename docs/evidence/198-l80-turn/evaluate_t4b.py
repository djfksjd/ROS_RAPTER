"""Repeat the decision-B full and extra gates for t4b only; no training."""
import hashlib
from pathlib import Path

import evaluate_decision_b as evaluation


if __name__ == '__main__':
    evaluation.RUNS = ('l80_t4b',)
    evaluation.OUT = Path(__file__).resolve().parent/'eval/t4b'
    original_hashes = evaluation.hashes

    def hashes():
        result = original_hashes()
        path = Path(__file__).resolve()
        result[str(path.relative_to(evaluation.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    evaluation.hashes = hashes
    evaluation.main()
