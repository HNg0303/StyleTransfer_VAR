import math
from functools import partial
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.style_vqvae import VQVAE, VectorQuantizer2
from models.cnn import VGG16Encoder, gram_matrix


FeatureTargets = Union[torch.Tensor, Mapping[str, torch.Tensor]]
StyleWeights = Union[float, Sequence[float], Mapping[str, float]]


def _target_like(target: torch.Tensor, actual: torch.Tensor, name: str) -> torch.Tensor:
    """Allow a single reference for the batch, but no channel/spatial broadcasting."""
    if target.ndim != actual.ndim or target.shape[1:] != actual.shape[1:]:
        raise ValueError(f'{name}: target shape {tuple(target.shape)} does not match {tuple(actual.shape)}.')
    if target.shape[0] not in (1, actual.shape[0]):
        raise ValueError(f'{name}: reference batch must be 1 or {actual.shape[0]}.')
    return target.expand_as(actual) # Expand single reference batch to actual batch size.


def _guidance_targets(content_features, style_features, gram_style_features,
                      content_layer, style_layers, style_alpha, device):
    """Detach reference graphs once, outside the repeatedly evaluated closure."""
    # Clone outside inference mode: inference tensors cannot be saved for backward.
    with torch.inference_mode(False), torch.no_grad(), torch.autocast(
        device_type=torch.device(device).type, enabled=False,
    ):
        content_target = None
        if content_features is not None:
            target = (content_features[content_layer]
                      if isinstance(content_features, Mapping) else content_features)
            content_target = target.detach().to(device=device, dtype=torch.float32).clone()

        source = gram_style_features if gram_style_features is not None else style_features
        if source is None:
            return content_target, {}, {}
        if isinstance(source, torch.Tensor):
            if style_layers is None or len(style_layers) != 1:
                raise ValueError('A single style tensor requires style_layers=("layer_name",).')
            source = {style_layers[0]: source}
        if style_layers is None:
            style_layers = tuple(style_alpha) if isinstance(style_alpha, Mapping) else tuple(source)
        else:
            style_layers = tuple(style_layers)
        if not style_layers or len(set(style_layers)) != len(style_layers):
            raise ValueError('style_layers must be nonempty and contain unique layer names.')
        if style_alpha is None:
            weights = dict.fromkeys(style_layers, 1.0 / len(style_layers))
        elif isinstance(style_alpha, Mapping):
            weights = {name: float(style_alpha[name]) for name in style_layers}
        elif isinstance(style_alpha, (float, int)):
            weights = dict.fromkeys(style_layers, float(style_alpha))
        else:
            if len(style_alpha) != len(style_layers):
                raise ValueError('style_alpha must have one weight per selected style layer; set style_layers explicitly.')
            weights = dict(zip(style_layers, map(float, style_alpha)))
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError('Style weights must be finite and nonnegative.')

        targets = {}
        for name in style_layers:
            target = source[name].detach().to(device=device, dtype=torch.float32).clone()
            if gram_style_features is None:
                target = gram_matrix(target)
            elif target.ndim == 2:
                # Accept a legacy unbatched Gram matrix for a single reference.
                target = target.unsqueeze(0)
            if target.ndim != 3 or target.shape[1] != target.shape[2]:
                raise ValueError(f'{name}: expected a Gram target of shape (B, C, C).')
            if weights[name] > 0:
                targets[name] = target
        return content_target, targets, {name: weights[name] for name in targets}


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class GramVAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0

        ## 0.a Define the embedding size of codebook (CVae) and Vocabulary Size of codebook.
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size

        ## 0.b. Depth: Number of AdaLNAttention blocks in the AR model.
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C) # Linear Projection from codebook embedding to VAR embedding
        
        # 2. class embedding: Content-specific generation and class embedding as the start token -> conditional embedding.
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V) # Linear projection from VAR embedding to codebook vocabulary logits.
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def _optimize_next_token_map(
        self, next_token_map, *, vgg_encoder, content_target, gram_style_targets,
        style_weights, content_layer, content_alpha, opti_steps, opti_lr,si, cutoff_idx, start_idx
    ):
        """Optimize context only; the cumulative decoder state is not modified.

        The context is a B,Cvae,pn_next,pn_next map. For a full-size perceptual
        comparison we bicubically upsample it to the VAE's maximum latent size.
        This is a decoder surrogate, not a loss through future AR predictions.
        """
        vae = self.vae_proxy[0]
        # Keep dropout/normalization deterministic, restoring caller-owned modes.
        modules = list(dict.fromkeys([*vae.modules(), *vgg_encoder.modules()]))
        modes = [module.training for module in modules]
        try:
            for module in modules:
                module.training = False
            with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
                device_type=next_token_map.device.type, enabled=False,
            ):
                latent = next_token_map.detach().float().clone().contiguous().requires_grad_(True)
                vae_dtype = next(vae.parameters(), latent).dtype
                encoder_dtype = next(vgg_encoder.parameters(), latent).dtype
                # LBFGS defaults to 20 inner iterations per step(). Restrict each
                # call to one evaluation/update attempt so opti_steps is explicit.
                optimizer = torch.optim.LBFGS(
                    [latent], lr=opti_lr, max_iter=1, max_eval=1,
                    tolerance_grad=0.0, tolerance_change=0.0,
                    history_size=10, line_search_fn=None,
                )

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    full_latent = F.interpolate(
                        latent, size=(self.patch_nums[-1], self.patch_nums[-1]), mode='bicubic',
                    )
                    decoded = vae.fhat_to_img(full_latent.to(dtype=vae_dtype))
                    if getattr(vgg_encoder, 'input_range', (0, 1)) == (0, 1):
                        decoded = (decoded + 1) * 0.5
                    encoded = vgg_encoder(decoded.to(dtype=encoder_dtype))
                    total_loss = latent.new_zeros(())
                    if content_target is not None and content_alpha > 0:
                        actual = encoded[content_layer].float()
                        target = _target_like(content_target, actual, f'content/{content_layer}')
                        total_loss = total_loss + content_alpha * F.mse_loss(actual, target)
                    for name, target in gram_style_targets.items():
                        actual = gram_matrix(encoded[name].float())
                        target = _target_like(target, actual, f'style/{name}')
                        total_loss = total_loss + style_weights[name] * F.mse_loss(actual, target)
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError('Non-finite perceptual guidance loss.')
                    # Gradients only for the optimized context: do not accumulate
                    # gradients on VAE/VGG parameters, even if they are trainable.
                    latent.grad = torch.autograd.grad(total_loss, latent)[0].contiguous()
                    if not torch.isfinite(latent.grad).all():
                        raise FloatingPointError('Non-finite perceptual guidance gradient.')

                    print(f'[resolution = {si}][opti_steps={opti_steps}][cutoff_idx={cutoff_idx}][start_idx={start_idx}] [closure] perceptual guidance loss={total_loss.item():.6f} (content={content_alpha:.3f}, style={sum(style_weights.values()):.3f})', flush=True)
                    return total_loss

                for _ in range(opti_steps):
                    optimizer.step(closure)
                if not torch.isfinite(latent).all():
                    raise FloatingPointError('Non-finite optimized next-token map.')
                return latent.detach().to(dtype=next_token_map.dtype)
        finally:
            for module, mode in zip(modules, modes):
                module.training = mode

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, opti_steps: int = 3, style_features: Optional[FeatureTargets] = None, content_features: Optional[FeatureTargets] = None, gram_style_features: Optional[FeatureTargets] = None,
        vgg_encoder: Optional[VGG16Encoder] = None,  # if None, will be initialized inside the function
        style_alpha: Optional[StyleWeights] = None, content_alpha: float = 0.5, returns_vemb=False,
        *, cutoff_idx: Optional[int] = None, start_idx: int = 0, opti_lr: float = 0.01,
        content_layer: str = 'relu4_3', style_layers: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param opti_steps: LBFGS update attempts per selected stage (one closure each);
            0 disables guidance. No model weights are optimized.
        :param cutoff_idx: exclusive cutoff over completed, zero-based AR stages:
            start_idx <= si < cutoff_idx. For example, 3 guides the inputs AFTER
            stages 0,1,2 (1x1,2x2,3x3). None selects all non-final stages.
            The final stage has no next-token input and cannot be selected.
        :param start_idx: inclusive start of the guidance window; defaults to 0.
        :param opti_lr: LBFGS learning rate, independent of stage placement.
        :param style_features: VGG feature dict; Gram matrices are computed once.
        :param gram_style_features: precomputed (B,C,C) Grams by VGG layer name;
            takes precedence over style_features. A single tensor requires one
            explicit style_layers entry. A reference batch of 1 can guide all B.
        :param content_features: VGG feature dict, or the tensor for content_layer.
            Reference spatial sizes must match the full decoded image's features.
        :param style_alpha: scalar, layer-name/weight dict, or sequence matching
            style_layers. None assigns equal weights summing to 1.
        :param style_layers: selected style layers; defaults to style_alpha keys
            for a weight dict, otherwise all supplied style-target layers.
        :param content_layer: spatial VGG layer for content MSE.
        :param vgg_encoder: reuse the SAME encoder/checkpoint as the references.
            If omitted, construct pretrained VGG16 once when guidance is active.
        :return: decoded images (B,3,H,W) in [0,1]. Guidance changes only context,
            not f_hat directly. No references/zero weights means ordinary VAR.
        """
        if not isinstance(opti_steps, int) or isinstance(opti_steps, bool) or opti_steps < 0:
            raise ValueError('opti_steps must be a nonnegative integer.')
        cutoff_idx = self.num_stages_minus_1 if cutoff_idx is None else cutoff_idx
        if (not isinstance(start_idx, int) or isinstance(start_idx, bool)
                or not isinstance(cutoff_idx, int) or isinstance(cutoff_idx, bool)
                or not 0 <= start_idx <= cutoff_idx <= self.num_stages_minus_1):
            raise ValueError(f'Require 0 <= start_idx <= cutoff_idx <= {self.num_stages_minus_1}; the final stage has no next-token map.')
        if not math.isfinite(opti_lr) or opti_lr <= 0:
            raise ValueError('opti_lr must be finite and positive.')
        if not math.isfinite(content_alpha) or content_alpha < 0:
            raise ValueError('content_alpha must be finite and nonnegative.')

        content_target, gram_targets, style_weights = None, {}, {}
        if opti_steps > 0 and start_idx < cutoff_idx:
            content_target, gram_targets, style_weights = _guidance_targets(
                content_features, style_features, gram_style_features,
                content_layer, style_layers, style_alpha, self.lvl_1L.device,
            )
        use_guidance = (content_target is not None and content_alpha > 0) or bool(gram_targets)
        if use_guidance:
            with torch.inference_mode(False):
                if vgg_encoder is None:
                    # Avoid perturbing the sampling RNG while constructing VGG.
                    with torch.random.fork_rng(devices=[]):
                        vgg_encoder = VGG16Encoder()
                vgg_encoder = vgg_encoder.to(self.lvl_1L.device)

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, dtype=torch.long, device=self.lvl_1L.device)
        else:
            label_B = label_B.to(device=self.lvl_1L.device, dtype=torch.long)
            label_B = torch.where(label_B < 0, self.num_classes, label_B)
        if label_B.shape != (B,) or ((label_B < 0) | (label_B > self.num_classes)).any():
            raise ValueError(f'label_B must have shape ({B},) with indices 0..{self.num_classes}, or -1 for unconditional.')
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0)) # B x C
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC # Level n: the nxn resolution that token belongs to and spatial position (exact i, j in h,w). 
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        try:
            for si, pn in enumerate(self.patch_nums):
                ratio = si / self.num_stages_minus_1
                cur_L += pn * pn
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                x = next_token_map
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)
                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                    si, len(self.patch_nums), f_hat, h_BChw,
                )
                if si != self.num_stages_minus_1:
                    if use_guidance and start_idx <= si < cutoff_idx:
                        next_token_map = self._optimize_next_token_map(
                            next_token_map, vgg_encoder=vgg_encoder,
                            content_target=content_target, gram_style_targets=gram_targets,
                            style_weights=style_weights, content_layer=content_layer,
                            content_alpha=content_alpha, opti_steps=opti_steps, opti_lr=opti_lr, si = si, cutoff_idx = cutoff_idx, start_idx = start_idx
                        )
                    # Only flatten AFTER decoder-based optimization.
                    next_token_map = next_token_map.reshape(B, self.Cvae, -1).transpose(1, 2)
                    next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)
        finally:
            for b in self.blocks:
                b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(GramVAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
