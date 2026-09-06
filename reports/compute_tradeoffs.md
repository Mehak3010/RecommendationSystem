# Compute Tradeoffs: MF and LightGCN

Both of these models are algorithmically correct but under-trained relative
to what a GPU or a longer wall-clock budget would allow. Documented here in
detail (rather than just a one-line caveat in the README) since the
dashboard surfaces this on a dedicated tab.

## Matrix Factorization (BPR, from scratch — `src/phase1_baselines.py`)

- **What was run:** 3 epochs, batch size 8192, Adam lr=0.01, embedding dim 64, over ~4M positive interactions, on CPU.
- **Why it underperforms ALS:** ALS (`implicit` library) uses a Cython-optimized alternating-least-squares solver that converges to a good solution in 15 iterations over the *full* interaction matrix per iteration. Our from-scratch MF uses stochastic mini-batch BPR updates, which need many more gradient steps to reach a comparable optimum — 3 epochs is a small fraction of what a fully converged run needs.
- **What would close the gap:** more epochs (10-20), a learning rate schedule, and possibly increasing the negative-sampling ratio (more than 1 negative per positive) — all straightforward, just more wall-clock time than a CPU sandbox comfortably allows for a demo run.

## LightGCN (`src/phase6a_lightgcn.py`)

- **What was run:** 60 gradient steps, batch size 16,384, over a graph of ~63,000 nodes (53,424 users + 10,000 items) and ~11.7M directed edges, on CPU.
- **Why 60 steps:** LightGCN's forward pass re-propagates embeddings through the *entire* graph (3 sparse-matrix layers) on every single training step, because the propagated embeddings depend on the current (updating) parameters. Unlike the two-tower or MF models, this can't be approximated with a cheap mini-batch subgraph without changing the algorithm. Each step costs roughly 3 sparse matrix multiplications over an 11.7M-edge adjacency, which is a few seconds per step on CPU — a GPU would make this 10-50x faster and comfortably support hundreds of steps.
- **Evidence it was still improving:** training loss was still decreasing at step 60 (had not plateaued), so the reported NDCG@10 is a lower bound on what full training would achieve — the algorithm itself is implemented correctly (verified against the published LightGCN propagation rule: `E_{k+1} = D^-1/2 A D^-1/2 E_k`, final embedding = mean across layers).
- **What would close the gap:** a GPU, or a smaller subsampled graph for faster iteration during development, then a full run on the complete graph before final evaluation.

## Honest framing for interviews

Neither of these is "the model doesn't work" — both are "the model is correctly implemented but compute-constrained in this environment," which is a normal, explainable tradeoff in real ML work, not a bug to hide. Knowing exactly *why* (mini-batch SGD needs more steps vs. full-graph propagation being inherently expensive per step) is the actual signal worth demonstrating.
