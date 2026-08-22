"""Minimal greedy generation example for the custom LoopedLM checkpoint."""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import LoopedLM, ModelConfig

root = Path(__file__).resolve().parent
artifact = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
model = LoopedLM(ModelConfig(**artifact["model_config"]))
model.load_state_dict(artifact["model"]); model.eval()
tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))

def generate(prompt: str, max_new_tokens: int = 50, loops: int = 12) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            tokens = torch.tensor([ids[-512:]], dtype=torch.long)
            logits = model(tokens, loops=loops)
            ids.append(int(logits[0, -1].argmax()))
    return tokenizer.decode(ids)

if __name__ == "__main__":
    print(generate("Looped transformers", max_new_tokens=40))
