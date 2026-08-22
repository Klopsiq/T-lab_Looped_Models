---
language: en
license: other
library_name: pytorch
tags:
- text-generation
- looped-transformer
- fineweb
---

# LoopedLM BPE8k, auxiliary anytime training

This is the final checkpoint of the Looped Models research project. It was selected before a single locked-test evaluation and was not replaced using post-test validation ablations.

- [Source code](https://github.com/Klopsiq/T-lab_Looped_Models)
- [Research report](https://github.com/Klopsiq/T-lab_Looped_Models/blob/main/REPORT.md)

The training dataset is FineWeb (ODC-By). No separate license has yet been selected for the model artifact.

- Architecture: two shared Qwen-style blocks, width 512, GQA 8/2, relative input injection, Depth-RoPE.
- Parameters: 9,440,513 unique parameters with tied input/output embeddings.
- Training: 24,969,216 FineWeb BPE tokens, context 512, depth sampled uniformly from 8 to 16, decaying intermediate LM loss.
- Selected inference depth: T=12.
- Locked test: NLL 4.2103, PPL 67.38 on 1,048,576 tokens from 1,251 documents.

The method improved validation and test quality inside the training-depth range, but extrapolation beyond T=16 degraded faster than the fixed-depth baseline. This checkpoint does not demonstrate useful test-time scaling to arbitrarily many loops.

Later validation-only ablations found that training over depths 8 to 24 shifts the best readout to T=16, and that final-only training beats the auxiliary objective on three of four seeds. Those findings are reported as future directions; they do not replace this checkpoint because they have no new locked-test result.

## Use

Install `requirements.txt` and run `python inference_example.py`. This repository uses custom PyTorch code rather than a Transformers `AutoModel` class. `model.pt` contains weights and the model configuration, without optimizer state. Exact tokenizer and data hashes are stored in `data_manifest.json`; measured results are included as JSON.

## Limitations

This is a 9.44M-parameter research model trained on one pinned FineWeb shard for about 25M tokens. It is not intended for factual use, instruction following, or deployment. The model card reports a single held-out test opening after all selection decisions.
