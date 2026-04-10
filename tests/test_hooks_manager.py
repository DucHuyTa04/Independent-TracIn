"""Tests for src/hooks_manager.py.

Validates:
    - Forward hook captures activation and logits
    - Context manager removes hooks on exit (even on exception)
    - 2D flattening: Linear (2D), Conv2d (4D), Transformer-like (3D)
    - Backward hook captures grad_output
"""

import pytest
import torch
import torch.nn as nn

from src.hooks_manager import HookManager, MultiLayerBackwardGhostManager, _flatten_to_2d


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
            # Linear(8, 4) with bias: activation is bias-augmented → (2, 9)
            assert hm.activation.shape == (2, 9)
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
            with pytest.raises(RuntimeError):
                _ = hm.activation

    def test_grad_output_error_in_forward_mode(self):
        model = self._make_model()
        with HookManager(model, model, backward=False) as hm:
            with pytest.raises(RuntimeError):
                _ = hm.grad_output


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

            # Linear(8, 4) with bias: activation is bias-augmented → (2, 9)
            assert hm.activation.shape == (2, 9)
            assert hm.grad_output.shape == (2, 4)


# ---- Hook test with CNN (4D tensors) ----


class TestHookManagerCNN:
    def test_cnn_flattening(self):
        """Conv2d input is im2col-unfolded and bias-augmented."""
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 8, 8)

        with HookManager(conv, conv) as hm:
            with torch.no_grad():
                out = conv(x)
            # Conv2d im2col: C_in*kH*kW = 3*3*3 = 27, + 1 bias → (2, 28)
            assert hm.activation.shape == (2, 28)
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
            # Activation is input to fc2 with bias augmentation: (2, 17)
            assert hm.activation.shape == (2, 17)
            assert hm.logits.shape == (2, 4)


class TestMultiLayerTorchBlocks:
    def test_torch_blocks_match_numpy(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 3)
                self.fc2 = nn.Linear(3, 2)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc2(h)

        model = Tiny()
        x = torch.randn(2, 4, requires_grad=True)
        y = torch.tensor([0, 1], dtype=torch.long)
        layers = [model.fc1, model.fc2]
        with MultiLayerBackwardGhostManager(layers) as hm:
            model.zero_grad(set_to_none=True)
            logits = model(x)
            nn.CrossEntropyLoss()(logits, y).backward()
            A_t, E_t = hm.torch_blocks()
            A_n, E_n = hm.numpy_blocks()
        assert len(A_t) == len(E_t) == 2
        for i in range(2):
            assert torch.allclose(A_t[i].cpu(), torch.from_numpy(A_n[i]))
            assert torch.allclose(E_t[i].cpu(), torch.from_numpy(E_n[i]))


# ---- ConvTranspose2d ghost vs autograd (per-layer weight inner product) ----


class _WrapCT(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def _conv_transpose_ghost_vec(layer: nn.ConvTranspose2d, x: torch.Tensor, target: torch.Tensor):
    w = _WrapCT(layer)
    with MultiLayerBackwardGhostManager([layer], keep_raw=True) as hm:
        w.zero_grad(set_to_none=True)
        out = w(x)
        nn.MSELoss(reduction="mean")(out, target).backward()
    ra, re = hm.raw_torch_blocks()
    g = torch.einsum("bth,btv->bhv", ra[0].float(), re[0].float()).reshape(-1)
    return g.double()


def _conv_transpose_auto_vec(layer: nn.ConvTranspose2d, x: torch.Tensor, target: torch.Tensor):
    w = _WrapCT(layer)
    w.zero_grad(set_to_none=True)
    out = w(x)
    nn.MSELoss(reduction="mean")(out, target).backward()
    parts = [layer.weight.grad.reshape(-1).double()]
    if layer.bias is not None:
        parts.append(layer.bias.grad.double())
    return torch.cat(parts)


class TestConvTranspose2dGhostMatchesAutograd:
    @pytest.mark.parametrize(
        "cin,cout,k,s,p,oph,opw,bias",
        [
            (3, 5, 3, 2, 1, 0, 0, True),
            (4, 4, 4, 2, 1, 0, 0, True),
            (2, 3, 3, 2, 0, 1, 0, False),
        ],
    )
    def test_ghost_inner_product_matches_weight_grad(
        self, cin, cout, k, s, p, oph, opw, bias,
    ):
        torch.manual_seed(0)
        ct = nn.ConvTranspose2d(
            cin,
            cout,
            k,
            stride=s,
            padding=p,
            output_padding=(oph, opw),
            bias=bias,
        )
        x1 = torch.randn(1, cin, 6, 6)
        x2 = torch.randn(1, cin, 6, 6)
        with torch.no_grad():
            o1 = ct(x1)
            t1 = torch.randn_like(o1)
            o2 = ct(x2)
            t2 = torch.randn_like(o2)

        g1 = _conv_transpose_ghost_vec(ct, x1, t1)
        g2 = _conv_transpose_ghost_vec(ct, x2, t2)
        a1 = _conv_transpose_auto_vec(ct, x1, t1)
        a2 = _conv_transpose_auto_vec(ct, x2, t2)

        dot_g = torch.dot(g1, g2).item()
        dot_a = torch.dot(a1, a2).item()
        assert dot_g == pytest.approx(dot_a, rel=1e-4, abs=1e-5)
        assert g1.norm().item() == pytest.approx(a1.norm().item(), rel=1e-4, abs=1e-5)


class TestBatchNorm2dEvalGhostMatchesAutograd:
    def test_gamma_beta_ghost_inner_product(self):
        torch.manual_seed(1)
        bn = nn.BatchNorm2d(4, affine=True)
        bn.eval()
        with torch.no_grad():
            bn.running_mean.uniform_(-0.2, 0.2)
            bn.running_var.uniform_(0.4, 1.2)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn = bn

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.bn(x)

        m = M()

        def ghost_vec(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with MultiLayerBackwardGhostManager([m.bn], keep_raw=True) as hm:
                m.zero_grad(set_to_none=True)
                out = m(x)
                nn.MSELoss(reduction="mean")(out, t).backward()
            ra, re = hm.raw_torch_blocks()
            return torch.cat([ra[0].float().reshape(-1), re[0].float().reshape(-1)]).double()

        def auto_vec(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            m.zero_grad(set_to_none=True)
            out = m(x)
            nn.MSELoss(reduction="mean")(out, t).backward()
            parts = [m.bn.weight.grad.reshape(-1).double(), m.bn.bias.grad.reshape(-1).double()]
            return torch.cat(parts)

        x1 = torch.randn(1, 4, 5, 5, requires_grad=True)
        x2 = torch.randn(1, 4, 5, 5, requires_grad=True)
        t1 = torch.randn_like(x1)
        t2 = torch.randn_like(x2)

        g1, g2 = ghost_vec(x1, t1), ghost_vec(x2, t2)
        a1, a2 = auto_vec(x1, t1), auto_vec(x2, t2)
        assert torch.dot(g1, g2).item() == pytest.approx(
            torch.dot(a1, a2).item(), rel=1e-4, abs=1e-5,
        )
