# GPT-2 from Scratch in NumPy

Angelina Huang's F26 ML Team Lab 3: the forward pass of GPT-2 small written from scratch in NumPy, first without attention and then with multi-head attention.

## Step 1: GPT-2 without attention

[gpt2_no_attention.py](gpt2_no_attention.py) adds:

- `embed`: looks up each token's embedding vector
- `add_positions`: adds each position's positional embedding
- `mlp`: the feed-forward layer, expanding to 4d with gelu and shrinking back to d
- `transformer_block`: layer norm, attention and MLP, each added to the residual stream
- `unembed`: turns each vector back into a score for every vocabulary token
- `forward`: the whole model, from token IDs to logits
- `softmax`: converts logits to probabilities, with temperature
- `sample`: picks the next token, greedy or random
- `generate`: produces text one token at a time

Attention is left as a placeholder that returns zeros, so each token cannot see the others.

## Step 2: GPT-2 with attention

[gpt2_with_attention.py](gpt2_with_attention.py) adds:

- `causal_mask`: blocks each token from attending to future tokens
- `attention`: scaled dot-product attention, `softmax(Q Kᵀ / √d_k) V`
- `split_heads` / `merge_heads`: split the 768-dim vectors into 12 heads of 64
- `multi_head_attention`: GPT-2's full attention sub-layer with the output projection
- `head_scores`: measures how much each head looks back a fixed number of tokens

With attention in place, the forward pass matches the official Hugging Face GPT-2.

### Findings

- **Previous-token head:** layer 4, head 11 puts all of its attention (score 1.00) on the token just before.
- **Induction heads:** on a random 25-token sequence repeated twice, loss drops from 12.70 on the first copy to 0.40 on the second. The heads that look back to the earlier copy are in layers 5–7 (top: layer 5, head 5 at 0.95).

## Running the tests

Requires Python 3 with NumPy, PyTorch, Hugging Face `transformers` and `tiktoken`. [lab_utils.py](lab_utils.py) loads GPT-2's weights and the tokenizer.

```bash
python3 test_gpt2_no_attention.py
python3 test_gpt2_with_attention.py
```
