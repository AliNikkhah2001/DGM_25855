# CHW-1 — MADE and gated PixelCNN

Coding half of Homework 1. Run the notebook or `chw1_impl.py` from this directory. The first run downloads MNIST into `data/`.

```bash
python3 chw1_impl.py
```

The notebook imports the same module. Training uses fused AdamW on MPS when that device is available. Progress bars use `tqdm.notebook` inside Jupyter and a text bar otherwise. Loss curves are written to `outputs/` after every epoch. TensorBoard logs go to `runs/` (created locally, not committed).

## What was trained

Splits are the real MNIST split: 60,000 train and 10,000 test. Test batches do not update weights.

**MADE.** Two hidden layers of 512, 10 epochs, masks redrawn every batch. Final test negative log-likelihood is 0.304 bits/dim. A uniform Bernoulli pixel is 1 bit/dim. Train ended at 0.306.

**Inpainting.** Observed pixels are held fixed. A hole that removes the ink has nothing to copy; a smaller hole with the border kept is filled from that border.

**FGSM.** Epsilon 0.3 drops the supplied CNN from 97.6% to 10.2% accuracy. After the 0.5 threshold the binary images barely move, so the MADE likelihood does not rise. It is not a detector for this attack.

**PixelCNN.** Five gated layers, 48 channels, 7×7 stacks, so the center pixel can see the whole past canvas. Six epochs of cross-entropy, then four more with ink pixels weighted twice as much as background. Causality was checked before training: the gradient at a pixel does not reach that pixel or anything below or to the right of it.

| | bits/dim |
|---|---|
| After 6 epochs, test | 0.170 |
| After the ink-weighted fine-tune, test | 0.172 |
| Uniform 4-level pixel | 2.0 |

The fine-tune spends a little likelihood on the background and cleans the strokes. On a fresh 10×10 grid, one row per class, the supplied CNN reads 91 of the 100 samples as the requested digit. The grid is `outputs/conditional_samples.png`. The earlier 6-epoch weights are `outputs/pixelcnn_e6.pt`.
