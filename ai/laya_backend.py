"""Laya (Apache-2.0) decision engine as a command backend.

Same typed `choice` question as the NanoJev backend; only the allowed action IDs
can come back. Weights are fetched from the Hugging Face Hub on first use at the
pinned revision in ai/models.json (public repo, no token needed).
"""
import json
from pathlib import Path
import time

from commands import ACTIONS, validate_text

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {'action': {'type': 'choice', 'instructions':
    'Which allowed action matches the operator command? Select REJECT if ambiguous, conflicting or unsupported.',
    'criteria': ACTIONS}}


class Laya:
    def __init__(self, device='mps', checkpoint='multilingual'):
        from laya import Router
        pin = json.loads((ROOT/'ai/models.json').read_text())['laya']
        self.router = Router(device=device, standalone_repos=True,
            revisions={checkpoint: pin['revisions'][checkpoint]}, preload=False)
        self.checkpoint = checkpoint

    def predict(self, text):
        text = validate_text(text)
        start = time.perf_counter()
        result = self.router.predict(text, QUESTION, model=self.checkpoint)
        answer = result['answers']['action']
        if answer['choice'] not in ACTIONS:
            raise ValueError('Unknown action')
        return {'action': answer['choice'], 'probabilities': answer.get('probabilities'),
            'confidence': answer.get('answer_confidence'),
            'backend': f'Laya-{self.checkpoint}', 'e2e_ms': (time.perf_counter()-start)*1000}
