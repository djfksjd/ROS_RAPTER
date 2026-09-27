"""Mac FP32 adapter for the actual upstream NanoJev decision model.

Uses the unchanged upstream DecisionModel, tokenization and probability decoder.
Only device setup differs from the upstream CUDA-only predictor.
"""
import json
from pathlib import Path
import sys
import time

from commands import ACTIONS, validate_text

ROOT = Path(__file__).resolve().parents[1]


class NanoJev:
    def __init__(self, device='mps', adapter=None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        sys.path.insert(0, str(ROOT / 'vendor/NanoJev/scripts'))
        from predict_toy_decisions import prepare_examples, answer_from_probabilities
        from train_toy_decisions import DecisionModel
        path = ROOT / 'models/NanoJev'
        config = json.loads((path / 'config.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer',
            local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(path / 'backbone_config',
            local_files_only=True, trust_remote_code=False)
        cfg.use_cache = False
        body = AutoModel.from_config(cfg, attn_implementation='sdpa', trust_remote_code=False)
        self.model = DecisionModel(body, config['set_head'])
        self.model.load_state_dict(load_file(str(path / 'best.safetensors')), strict=True)
        if adapter:
            head = load_file(str(adapter))
            expected = {k for k in self.model.state_dict() if not k.startswith('backbone.')}
            if set(head) != expected:
                raise ValueError('Adapter must contain exactly the decision head parameters')
            self.model.load_state_dict(head, strict=False)
        self.model.to(device=device, dtype=torch.float32).eval()
        self.torch, self.device = torch, device
        self.prepare, self.answer = prepare_examples, answer_from_probabilities
        self.max_length = config.get('max_length', 512)
        self.adapter = bool(adapter)

    def predict(self, text):
        text = validate_text(text)
        start = time.perf_counter()
        payload = {'states': [{'id': 'operator', 'state': text, 'questions': {
            'action': {'type': 'choice', 'instructions':
                'Which allowed action matches the operator command? Select REJECT if ambiguous, conflicting or unsupported.',
                'criteria': ACTIONS}}}]}
        examples = self.prepare(payload, self.tokenizer, self.max_length)
        if self.device == 'mps':
            self.torch.mps.synchronize()
        forward = time.perf_counter()
        with self.torch.inference_mode():
            logits, _ = self.model(examples, self.tokenizer.pad_token_id)
            probs = logits[0, :len(ACTIONS)].float().softmax(-1).cpu().tolist()
        model_ms = (time.perf_counter()-forward)*1000
        result = self.answer(examples[0], probs)
        return {'action': result['choice'], 'probabilities': result['probabilities'],
            'backend': 'NanoJev-raptor-head-mps-fp32' if self.adapter else 'NanoJev-unified-games-v1-mps-fp32',
            'e2e_ms': (time.perf_counter()-start)*1000, 'model_ms': model_ms}
