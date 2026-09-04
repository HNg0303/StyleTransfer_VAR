"""Offline CPU tests: python -m unittest discover -s test -p test_style_var_guidance.py -v"""

import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from models.cnn import VGG16Encoder, gram_matrix
from models.quant import VectorQuantizer2
from models.style_var import GramVAR
from models.style_vqvae import VQVAE
from models.var import VAR


class TinyVAE(nn.Module):
    Cvae = 3
    vocab_size = 11

    def __init__(self, patch_nums=(1, 2, 3, 4)):
        super().__init__()
        self.quantize = VectorQuantizer2(
            self.vocab_size, self.Cvae, False, v_patch_nums=patch_nums, share_quant_resi=1,
        )
        self.post_quant_conv = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.post_quant_conv.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1) * 0.25)
            self.post_quant_conv.bias.zero_()
        self.decoder = nn.Sequential(nn.Upsample(size=(16, 16), mode='bilinear'), nn.Tanh())

    def fhat_to_img(self, latent):
        if latent.ndim != 4:
            raise AssertionError('The decoder must receive a 4D latent.')
        if latent.shape[-2:] != (self.quantize.v_patch_nums[-1],) * 2:
            raise AssertionError('The loss decoder must receive the full latent resolution.')
        return self.decoder(self.post_quant_conv(latent))


class TinyEncoder(nn.Module):
    input_range = (0, 1)

    def __init__(self):
        super().__init__()
        # Intentionally trainable: guidance must not accumulate parameter grads.
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        image = image * self.gain
        return {
            'relu1_2': image,
            'relu2_2': F.avg_pool2d(image, 2),
            'relu4_3': F.avg_pool2d(image, 4),
        }


def make_var(vae, cls=GramVAR):
    with contextlib.redirect_stdout(io.StringIO()):
        model = cls(
            vae, num_classes=5, depth=1, embed_dim=12, num_heads=3,
            patch_nums=vae.quantize.v_patch_nums, flash_if_available=False,
            fused_if_available=False,
        )
        model.init_weights()
    return model.eval()


class GramTests(unittest.TestCase):
    def test_batch_items_are_independent(self):
        features = torch.arange(48.0).reshape(2, 3, 2, 4)
        result = gram_matrix(features)
        self.assertEqual(result.shape, (2, 3, 3))
        for i in range(2):
            flattened = features[i].reshape(3, 8)
            torch.testing.assert_close(result[i], flattened @ flattened.T / 24)
            torch.testing.assert_close(result[i:i+1], gram_matrix(features[i:i+1]))

    def test_noncontiguous_features_and_gradients(self):
        features = torch.randn(2, 3, 2, 4, requires_grad=True)
        gram_matrix(features.transpose(-1, -2)).square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        with self.assertRaises(ValueError):
            gram_matrix(torch.zeros(3, 4))

    def test_gram_accumulates_in_float32_under_autocast(self):
        features = torch.full((2, 3, 32, 32), 100.0, dtype=torch.float16)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = gram_matrix(features)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, torch.full((2, 3, 3), 10000.0 / 3))


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.vae = TinyVAE()
        self.model = make_var(self.vae)
        self.encoder = TinyEncoder()
        with torch.no_grad():
            self.content = self.encoder(torch.full((1, 3, 16, 16), 0.8))
            self.style = self.encoder(torch.rand(1, 3, 16, 16) * 0.2)
        self.encoder.calls = 0

    def run_model(self, **kwargs):
        args = dict(
            B=2, label_B=-1, g_seed=42, cfg=0.0,
            content_features=self.content, style_features=self.style,
            vgg_encoder=self.encoder, opti_steps=3,
        )
        args.update(kwargs)
        return self.model.autoregressive_infer_cfg(**args)

    def helper_args(self, **kwargs):
        args = dict(
            vgg_encoder=self.encoder, content_target=self.content['relu4_3'],
            gram_style_targets={}, style_weights={}, content_layer='relu4_3',
            content_alpha=1.0, opti_steps=3, opti_lr=0.1,
        )
        args.update(kwargs)
        return args

    def test_cutoff_is_exclusive_and_final_stage_is_never_optimized(self):
        for start, cutoff, expected_pn in (
            (0, 2, [2, 3]), (1, 3, [3, 4]), (0, None, [2, 3, 4]),
            (0, 0, []), (2, 2, []),
        ):
            with self.subTest(start=start, cutoff=cutoff):
                with patch.object(self.model, '_optimize_next_token_map', side_effect=lambda x, **kw: x) as optimize:
                    output = self.run_model(start_idx=start, cutoff_idx=cutoff)
                self.assertEqual([call.args[0].shape[-1] for call in optimize.call_args_list], expected_pn)
                self.assertTrue(all(call.kwargs['opti_steps'] == 3 for call in optimize.call_args_list))
                self.assertEqual(output.shape, (2, 3, 16, 16))

    def test_disabled_guidance_matches_original_var_without_constructing_vgg(self):
        baseline = make_var(self.vae, VAR)
        baseline.load_state_dict(self.model.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            expected = baseline.autoregressive_infer_cfg(B=2, label_B=-1, g_seed=42, cfg=0.0)
        for kwargs in (
            {'opti_steps': 0}, {'cutoff_idx': 0},
            {'content_features': None, 'style_features': None},
            {'content_alpha': 0.0, 'style_alpha': 0.0},
        ):
            with self.subTest(kwargs=kwargs), patch('models.style_var.VGG16Encoder') as constructor:
                actual = self.run_model(vgg_encoder=None, **kwargs)
                constructor.assert_not_called()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_context_changes_but_source_weights_and_parameter_grads_do_not(self):
        context = torch.zeros(2, 3, 2, 2)
        original = context.clone()
        params = list(self.vae.parameters()) + list(self.encoder.parameters())
        weights = [p.detach().clone() for p in params]
        flags = [p.requires_grad for p in params]
        modes = [m.training for m in [*self.vae.modules(), *self.encoder.modules()]]
        refined = self.model._optimize_next_token_map(context, **self.helper_args())
        self.assertEqual(self.encoder.calls, 3)
        self.assertGreater((refined - original).abs().sum().item(), 0)
        self.assertFalse(refined.requires_grad)
        torch.testing.assert_close(context, original)
        for p, weight, flag in zip(params, weights, flags):
            torch.testing.assert_close(p, weight, rtol=0, atol=0)
            self.assertEqual(p.requires_grad, flag)
            self.assertIsNone(p.grad)
        self.assertEqual(modes, [m.training for m in [*self.vae.modules(), *self.encoder.modules()]])

    def test_lbfgs_evaluates_once_per_step_and_weights_distinct_style_layers(self):
        context = torch.zeros(2, 3, 2, 2)
        targets = {name: gram_matrix(self.style[name]) for name in ('relu1_2', 'relu2_2')}
        weights = {'relu1_2': 0.2, 'relu2_2': 0.8}
        with torch.no_grad():
            features = self.encoder((self.vae.fhat_to_img(F.interpolate(context, (4, 4), mode='bicubic')) + 1) / 2)
            expected = sum(weights[name] * F.mse_loss(gram_matrix(features[name]), target.expand(2, -1, -1)) for name, target in targets.items())
        losses = []
        original_step = torch.optim.LBFGS.step

        def counted_step(optimizer, closure):
            def counted_closure():
                loss = closure()
                losses.append(loss.item())
                return loss
            return original_step(optimizer, counted_closure)

        with patch.object(torch.optim.LBFGS, 'step', counted_step):
            self.model._optimize_next_token_map(context, **self.helper_args(
                content_target=None, gram_style_targets=targets, style_weights=weights,
            ))
        self.assertEqual(len(losses), 3)
        self.assertAlmostEqual(losses[0], expected.item(), places=7)

    def test_guidance_reduces_surrogate_content_loss(self):
        context = torch.zeros(2, 3, 2, 2)

        def loss(value):
            image = (self.vae.fhat_to_img(F.interpolate(value, (4, 4), mode='bicubic')) + 1) / 2
            return (self.encoder(image)['relu4_3'] - self.content['relu4_3']).square().mean()

        before = loss(context).item()
        refined = self.model._optimize_next_token_map(context, **self.helper_args())
        self.assertLess(loss(refined).item(), before)

    def test_outer_inference_mode_autocast_and_inference_reference_tensors(self):
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            content = self.encoder(torch.rand(1, 3, 16, 16))
            style = self.encoder(torch.rand(1, 3, 16, 16))
            actual = self.run_model(content_features=content, style_features=style, cutoff_idx=2)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))
        self.assertFalse(self.model.blocks[0].attn.caching)

    def test_optimization_does_not_directly_modify_the_decoded_accumulator(self):
        def fixed_sample(logits, **kwargs):
            return torch.zeros(*logits.shape[:2], 1, dtype=torch.long, device=logits.device)

        contexts = []
        handle = self.model.word_embed.register_forward_pre_hook(lambda module, args: contexts.append(args[0].detach().clone()))
        try:
            with patch('models.style_var.sample_with_top_k_top_p_', side_effect=fixed_sample):
                baseline = self.run_model(opti_steps=0)
                guided = self.run_model(cutoff_idx=2, opti_lr=0.1)
        finally:
            handle.remove()
        torch.testing.assert_close(guided, baseline, rtol=0, atol=0)
        self.assertGreater((contexts[0] - contexts[3]).abs().sum().item(), 0)

    def test_reference_dicts_grams_and_single_tensor_paths_agree(self):
        grams = {name: gram_matrix(value) for name, value in self.style.items()}
        actual = self.run_model(cutoff_idx=1)
        expected = self.run_model(cutoff_idx=1, gram_style_features=grams)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The old helper's B=1 unbatched Gram is accepted with an explicit layer.
        actual = self.run_model(
            cutoff_idx=1, content_features=self.content['relu4_3'],
            gram_style_features=grams['relu2_2'][0], style_layers=('relu2_2',), style_alpha=(0.5,),
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_reference_graphs_are_detached_and_encoder_is_constructed_once(self):
        reference = torch.rand(1, 3, 16, 16, requires_grad=True)
        features = self.encoder(reference)
        with patch('models.style_var.VGG16Encoder', return_value=self.encoder) as constructor:
            self.run_model(vgg_encoder=None, content_features=features, style_features=features, cutoff_idx=2)
        constructor.assert_called_once_with()
        self.assertIsNone(reference.grad)
        self.assertIsNone(self.encoder.gain.grad)

    def test_unconditional_tensor_label_and_repeated_runs(self):
        scalar = self.run_model(cutoff_idx=1)
        tensor = self.run_model(cutoff_idx=1, label_B=torch.tensor([-1, -1]))
        torch.testing.assert_close(scalar, tensor, rtol=0, atol=0)

    def test_invalid_schedules_and_weights(self):
        for kwargs in (
            {'opti_steps': -1}, {'opti_steps': 1.5}, {'cutoff_idx': 4},
            {'cutoff_idx': -1}, {'start_idx': 2, 'cutoff_idx': 1},
            {'opti_lr': 0}, {'content_alpha': -1}, {'style_alpha': [-1, 1, 1]},
            {'style_alpha': [0.5, 0.5]}, {'label_B': torch.tensor([0, 6])},
            {'style_features': self.style['relu1_2']},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_model(**kwargs)

    def test_shape_errors_clear_kv_cache_and_restore_modes(self):
        modes = [m.training for m in self.encoder.modules()]
        with self.assertRaisesRegex(ValueError, 'target shape'):
            self.run_model(content_features=torch.zeros(1, 3, 2, 2), cutoff_idx=1)
        self.assertEqual(modes, [m.training for m in self.encoder.modules()])
        for block in self.model.blocks:
            self.assertFalse(block.attn.caching)
            self.assertIsNone(block.attn.cached_k)
            self.assertIsNone(block.attn.cached_v)

    def test_native_negative_one_to_one_encoder_input_range(self):
        class NegativeRangeEncoder(TinyEncoder):
            input_range = (-1, 1)

            def forward(self, image):
                return super().forward((image + 1) / 2)

        reference = self.run_model(cutoff_idx=2)
        actual = self.run_model(cutoff_idx=2, vgg_encoder=NegativeRangeEncoder())
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_more_smooth_branch(self):
        image = self.run_model(more_smooth=True, cutoff_idx=1)
        self.assertTrue(torch.isfinite(image).all())

    def test_real_vae_and_vgg_architectures_have_a_latent_gradient(self):
        # Real network operators, small VAE, random VGG weights: no downloads.
        vae = VQVAE(ch=32, z_channels=3, vocab_size=11, v_patch_nums=(1, 2), share_quant_resi=1)
        model = make_var(vae)
        encoder = VGG16Encoder(weights=None)
        with torch.inference_mode():
            target_features = encoder(torch.rand(1, 3, 32, 32))
            # Exercise normal target preparation so inference tensors are cloned.
            image = model.autoregressive_infer_cfg(
                B=1, label_B=-1, g_seed=7, cfg=0.0, cutoff_idx=1, opti_steps=1,
                content_features=target_features, content_layer='relu1_2',
                style_features=target_features, style_alpha={'relu2_2': 0.5},
                vgg_encoder=encoder,
            )
        self.assertEqual(image.shape, (1, 3, 32, 32))
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(all(p.grad is None for p in vae.parameters()))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_existing_optional_content_addition_remains_unchanged(self):
        quant = self.vae.quantize
        current = torch.rand(2, 3, 1, 1)
        content = [torch.zeros(2, pn * pn, dtype=torch.long) for pn in (1, 2, 3, 4)]
        embedded_content = quant.embedding(content[0]).transpose(1, 2).reshape(2, 3, 1, 1)
        expected = quant.quant_resi[0.0](F.interpolate(current + embedded_content, (4, 4), mode='bicubic'))
        actual, context = quant.get_next_autoregressive_input(0, 4, torch.zeros(2, 3, 4, 4), current, content)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(context, F.interpolate(expected, (2, 2), mode='area'))


if __name__ == '__main__':
    unittest.main()
