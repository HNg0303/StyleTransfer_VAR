# GramVAR context-guidance experiments

`GramVAR.autoregressive_infer_cfg` optimizes the **next-token conditioning map**,
not the model weights or the accumulated `f_hat`. This preserves the intended
context-only experiment. It is not cumulative-latent refinement or optimization
of the current scale's sampled code embeddings.

## Notebook example

Assume `vae` and `var` have already been constructed with `build_style_vae_var`,
loaded from their checkpoints, and moved to `device`. `var` must be a `GramVAR`,
not the original `VAR`. After editing Python modules, restart/reload the notebook
and rebuild/reload model instances before testing.

```python
import torch
from models.cnn import VGG16Encoder, gram_matrix

vae.eval()
var.eval()

# Content/style tensors are RGB BCHW in [-1, 1]. Content image resolution must
# match the final generated image resolution. The style reference can differ.
vgg = VGG16Encoder(input_range=(-1, 1)).to(device).eval()
with torch.no_grad():
    content_features = vgg(content_img)
    style_features = vgg(style_img)
    style_grams = {name: gram_matrix(value.float())
                   for name, value in style_features.items()}

result = var.autoregressive_infer_cfg(
    B=content_img.shape[0],
    label_B=-1,              # unconditional embedding; tensor -1 is also handled
    cfg=0.0,
    g_seed=42,
    top_k=900,
    top_p=0.95,
    vgg_encoder=vgg,        # reuse the same checkpoint used for the references
    content_features=content_features,
    gram_style_features=style_grams,
    content_layer="relu4_3",
    content_alpha=0.5,
    style_alpha={"relu1_2": 0.5, "relu2_2": 0.5},
    opti_steps=3,
    opti_lr=0.01,
    start_idx=0,
    cutoff_idx=3,
)
```

Alternatively, pass `style_features=style_features` and omit `gram_style_features`;
the Gram matrices are computed once internally. A single reference image can
guide a larger generation batch. Gram matrices are computed independently for
each image, with shape `(B, C, C)`.

The loss is content MSE at `content_layer`, plus a weighted sum of distinct
per-layer Gram MSE losses. No pixel-reconstruction or latent-anchor loss is added.
The example weights are starting values, not validated optimal hyperparameters.

## Stage selection

The condition is `start_idx <= si < cutoff_idx`, using **zero-based completed AR
stage indices**. Guidance occurs after that stage has been sampled and accumulated,
before `word_embed` prepares the following stage's input.

For `patch_nums=(1,2,3,4,5,6,8,10,13,16)`:

| Experiment | `start_idx` | `cutoff_idx` | `opti_steps` |
|---|---:|---:|---:|
| Disabled baseline | 0 | 0 | 0 |
| After early stages `si=0,1,2` | 0 | 3 | 3 |
| After middle stages `si=3,4,5` | 3 | 6 | 3 |
| After late stages `si=6,7,8` | 6 | 9 | 3 |
| After all non-final stages | 0 | 9 | 1 |

For example, `si=0` has just generated the `1x1` code map; its optimized **next
input is `2x2`**. `cutoff_idx=None` selects all non-final stages. The final stage
has no next input, so `cutoff_idx=10` and final-only refinement are not supported
by this context-only method. SOS is not optimized.

`opti_steps=0`, an empty stage window, no references, or all-zero active loss
weights bypass optimization and do not instantiate VGG. The original quantizer's
explicit content-token addition remains available to other callers, but GramVAR
does not apply that addition implicitly.

## Optimizer and loss details

- Each selected stage creates a fresh LBFGS optimizer. `max_iter=1`,
  `max_eval=1`, and no line search make each `opti_steps` iteration one closure
  evaluation/update attempt. A zero gradient can still produce a no-op.
- The optimized variable is a detached, cloned, float32 BCHW context tensor.
  It is upsampled to the maximum latent size before VAE decoding, so perceptual
  losses compare full-size images. This decoded surrogate is **not** the actual
  future AR output; improving its loss does not guarantee improved final style.
- Local gradient and inference-mode overrides support callers using
  `torch.no_grad()`, `torch.inference_mode()`, and AMP/autocast. Optimizer/loss
  calculations disable autocast. Standard float32 model weights are recommended.
- Reference features are detached/cloned once. Gradients are requested only for
  the context tensor, not VAE/VGG parameters. Decoder/encoder evaluation modes are
  temporary and restored; KV caches are cleared even if guidance fails.
- `style_alpha` can be a layer-weight dictionary, a scalar, or a sequence with
  one weight per selected `style_layers`. With no weights, all provided style
  layers receive equal weights summing to one. With only two sequence weights,
  also pass `style_layers=("relu1_2", "relu2_2")`.
- A single style-feature/Gram tensor requires one explicit `style_layers` name.
  A single content-feature tensor is interpreted as `content_layer`.
- Precompute references again after updating `gram_matrix`: the old function
  mixed images across the batch and returned `(B*C, B*C)`.

## Offline verification

```text
python -m unittest discover -s test -p "test_*.py" -v
```

The tests cover cutoff boundaries, exact closure counts, baseline equivalence,
per-layer losses, batching, inference-mode/autocast gradients, cache cleanup,
context-only state changes, and real VAE/VGG operators with random offline
weights. They do not establish pretrained stylization quality or CUDA behavior.
