"""Offline checks: python -m unittest discover -s test -p test_cnn_encoder.py -v"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights

from models.cnn import Normalization, VGG16Encoder


class NormalizationTests(unittest.TestCase):
    def test_imagenet_statistics_and_input_are_not_mutated(self):
        norm = Normalization()
        image = torch.rand(2, 3, 5, 7)
        original = image.clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        torch.testing.assert_close(norm(image), (image - mean) / std)
        torch.testing.assert_close(image, original)
        self.assertEqual(set(norm.state_dict()), {'mean', 'std'})
        self.assertEqual(list(norm.parameters()), [])

    def test_dtype_and_input_gradients(self):
        norm = Normalization().double()
        image = torch.rand(1, 3, 4, 4, dtype=torch.float64, requires_grad=True)
        result = norm(image)
        self.assertEqual(result.dtype, torch.float64)
        result.sum().backward()
        torch.testing.assert_close(image.grad, norm.std.reciprocal().expand_as(image))

    def test_invalid_statistics_and_images(self):
        for kwargs in (
            {'mean': (0.5,)},
            {'std': (0.2, 0.0, 0.2)},
            {'std': (0.2, -0.1, 0.2)},
            {'mean': (float('nan'), 0.5, 0.5)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Normalization(**kwargs)
        for shape in ((3, 8, 8), (1, 1, 8, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                Normalization()(torch.zeros(shape))
        with self.assertRaises(TypeError):
            Normalization()(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))


class VGG16EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        # The real VGG16 architecture, without a network download.
        cls.encoder = VGG16Encoder(weights=None)

    @classmethod
    def tearDownClass(cls):
        del cls.encoder
        torch.set_num_threads(cls.original_threads)

    def test_feature_names_shapes_and_frozen_weights(self):
        image = torch.rand(2, 3, 32, 48)
        with torch.no_grad():
            features = self.encoder(image)
        expected = {
            'relu1_2': (2, 64, 32, 48),
            'relu2_2': (2, 128, 16, 24),
            'relu3_3': (2, 256, 8, 12),
            'relu4_3': (2, 512, 4, 6),
            'relu5_3': (2, 512, 2, 3),
        }
        self.assertEqual(list(features), list(expected))
        for name, shape in expected.items():
            self.assertEqual(tuple(features[name].shape), shape)
        self.assertFalse(self.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in self.encoder.parameters()))
        self.assertFalse(any(isinstance(layer, nn.Linear) for layer in self.encoder.modules()))

    def test_gradients_reach_image_but_not_frozen_weights(self):
        image = torch.rand(1, 3, 32, 32, requires_grad=True)
        features = self.encoder(image)
        sum(feature.square().mean() for feature in features.values()).backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))

    def test_var_input_range_matches_zero_one_input(self):
        # Share the same VGG weights to isolate the range conversion.
        with patch('models.cnn.vgg16', return_value=SimpleNamespace(features=self.encoder.features)):
            var_encoder = VGG16Encoder(weights=None, input_range=(-1, 1))
        image = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            expected = self.encoder(image)
            actual = var_encoder(image * 2 - 1)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_default_requests_pretrained_weights_without_downloading(self):
        fake_vgg = SimpleNamespace(features=nn.Sequential(nn.Conv2d(3, 3, 1)))
        with patch('models.cnn.vgg16', return_value=fake_vgg) as loader:
            encoder = VGG16Encoder(progress=False)
        loader.assert_called_once_with(weights=VGG16_Weights.IMAGENET1K_V1, progress=False)
        torch.testing.assert_close(encoder.normalization.mean, Normalization().mean)
        torch.testing.assert_close(encoder.normalization.std, Normalization().std)

    def test_alternate_checkpoint_statistics_and_trainable_option(self):
        weights = VGG16_Weights.IMAGENET1K_FEATURES
        fake_vgg = SimpleNamespace(features=nn.Sequential(nn.Conv2d(3, 3, 1)))
        with patch('models.cnn.vgg16', return_value=fake_vgg):
            encoder = VGG16Encoder(weights=weights, freeze=False)
        expected_std = torch.tensor(weights.transforms().std).view(1, 3, 1, 1)
        torch.testing.assert_close(encoder.normalization.std, expected_std)
        self.assertTrue(all(p.requires_grad for p in encoder.features.parameters()))

    def test_invalid_input_range_and_small_images(self):
        with self.assertRaises(ValueError):
            VGG16Encoder(weights=None, input_range=(0, 255))
        with self.assertRaises(ValueError):
            self.encoder(torch.rand(1, 3, 15, 32))


if __name__ == '__main__':
    unittest.main()
