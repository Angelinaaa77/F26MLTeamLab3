# F26 ML Team Lab 3

Angelina Huang's Lab 3: GPT-2 small written from scratch in NumPy.

## Part 1: forward pass without attention

[lab3_part1_angelina_huang.py](lab3_part1_angelina_huang.py) covers embeddings, positional embeddings, the MLP, transformer blocks, unembedding, softmax with temperature, sampling and text generation. Attention is left as a placeholder that returns zeros.

## Part 2: attention

[lab3_part2_angelina_huang.py](lab3_part2_angelina_huang.py) adds:

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

The test files import the unnamed module (`lab3_part1` / `lab3_part2`). To run them, change the module name in `test_part1.py` / `test_part2.py` to the `_angelina_huang` file, then run `python3 test_part1.py` or `python3 test_part2.py`.
