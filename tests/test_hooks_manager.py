"""Tests for src/hooks_manager.py.

Validates:
    - Forward hook captures activation and logits
    - Context manager removes hooks on exit (even on exception)
    - 2D flattening: Linear (2D), Conv2d (4D), Transformer-like (3D)
    - Backward hook captures grad_output
"""

import torch
import torch.nn as nn

from src.hooks_manager import HookManager, _flatten_to_2d


# ---- Flattening tests ----


class TestFlattenTo2D:
    def test_2d_passthrough(self):
        t = torch.randn(4, 16)
        out = _flatten_to_2d(t)
        assert out.shape == (4, 16)
        assert torch.equal(t, out)

    def test_3d_transformer(self):
        # [Batch, Seq, Hidden] -> mean over seq -> [Batch, Hidden]
        t = torch.randn(4, 10, 16)
        out = _flatten_to_2d(t)
        assert out.shape == (4, 16)
        expected = t.mean(dim=1)
        assert torch.allclose(out, expected)

    def test_4d_cnn(self):
        # [Batch, Channels, H, W] -> mean over spatial -> [Batch, Channels]
        t = torch.randn(4, 32, 8, 8)
        out = _flatten_to_2d(t)
        assert out.shape == (4, 32)
        expected = t.mean(dim=(2, 3))
        assert torch.allclose(out, expected)

    def test_unsupported_5d_raises(self):
        t = torch.randn(2, 3, 4, 5, 6)
        try:
            _flatten_to_2d(t)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ---- Hook tests with Linear model ----


class TestHookManagerLinear:
    def _make_model(self):
        model = nn.Linear(8, 4)
        return model

    def test_forward_captures_activation(self):
        model = self._make_model()
        x = torch.randn(2, 8)
        with HookManager(model, model) as hm:
            with torch.no_grad():
                logits = model(x)
            assert hm.activation.shape == (2, 8)
            assert hm.logits.shape == (2, 4)
            assert torch.allclose(logits, hm.logits)

    def test_hooks_removed_after_exit(self):
        model = self._make_model()
        with HookManager(model, model) as hm:
            pass
        # After exit, hooks should be removed
        assert len(hm._handles) == 0

    def test_hooks_removed_on_exception(self):
        model = self._make_model()
        hm = HookManager(model, model)
        try:
            with hm:
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        assert len(hm._handles) == 0

    def test_activation_error_before_forward(self):
        model = self._make_model()
        with HookManager(model, model) as hm:
            try:
                _ = hm.activation
                assert False, "Should have raised RuntimeError"
            except RuntimeError:
                pass

    def test_grad_output_error_in_forward_mode(self):
        model = self._make_model()
        with HookManager(model, model, backward=False) as hm:
            try:
                _ = hm.grad_output
                assert False, "Should have raised RuntimeError"
            except RuntimeError:
                pass


# ---- Backward hook test ----


class TestHookManagerBackward:
    def test_backward_captures_grad_output(self):
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8)
        targets = torch.randn(2, 4)

        with HookManager(model, model, backward=True) as hm:
            logits = model(x)
            loss = nn.MSELoss()(logits, targets)
            loss.backward()

            assert hm.activation.shape == (2, 8)
            assert hm.grad_output.shape == (2, 4)


# ---- Hook test with CNN (4D tensors) ----


class TestHookManagerCNN:
    def test_cnn_flattening(self):
        """Conv2d output is 4D [B, C, H, W] -> hook flattens to [B, C]."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8)

        with HookManager(conv, conv) as hm:
            with torch.no_grad():
                out = conv(x)
            # Conv input is 4D [2, 3, 8, 8] -> flattened to [2, 3]
            assert hm.activation.shape == (2, 3)
            assert hm.activation.dim() == 2


# ---- Hook test with multi-layer model (hook specific layer) ----


class TestHookManagerMultiLayer:
    def test_hook_specific_layer(self):
        """Hook only the last linear layer in a multi-layer model."""

        class TwoLayerMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = TwoLayerMLP()
        x = torch.randn(2, 8)

        with HookManager(model, model.fc2) as hm:
            with torch.no_grad():
                logits = model(x)
            # Activation should be the input to fc2: shape (2, 16)
            assert hm.activation.shape == (2, 16)
            assert hm.logits.shape == (2, 4)
