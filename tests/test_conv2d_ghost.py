"""Tests for exact Conv2d ghost via sum-over-spatial (3D raw hooks)."""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch
import torch.nn as nn

from benchmarks.ghost_faiss import (
    _ghost_matrix_torch_from_raw_blocks,
    _layer_ghost_dots_from_raw_blocks,
    _run_forward_backward,
)
from src.hooks_manager import MultiLayerBackwardGhostManager


def _sum_loss(pred: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return pred.sum()


class TestConv2dGhostMatchesAutogradDirection:
    """Per-sample conv weight gradient vs materialised ghost (exact vec match, no /L scaling)."""

    def test_single_conv_cosine_one(self):
        torch.manual_seed(0)
        conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
        x1 = torch.randn(1, 3, 8, 8)
        x2 = torch.randn(1, 3, 8, 8)

        model = nn.Sequential(conv)
        model.train()

        def get_ghost(x):
            with MultiLayerBackwardGhostManager([conv], keep_raw=True) as hm:
                model.zero_grad(set_to_none=True)
                y = model(x)
                _sum_loss(y, torch.empty(0)).backward()
                raw_a, raw_e = hm.raw_torch_blocks()
            return _ghost_matrix_torch_from_raw_blocks(raw_a, raw_e)[0]

        def get_auto(x):
            model.zero_grad(set_to_none=True)
            y = model(x)
            _sum_loss(y, torch.empty(0)).backward()
            parts = [conv.weight.grad.reshape(-1)]
            if conv.bias is not None:
                parts.append(conv.bias.grad.reshape(-1))
            return torch.cat(parts).float()

        g1 = get_ghost(x1)
        g2 = get_ghost(x2)
        a1 = get_auto(x1)
        a2 = get_auto(x2)

        # Ghost and autograd use different flattening orders (bhv vs vh), but
        # inner products are permutation-invariant: <g_q, g_t> == <a_q, a_t>
        ghost_dot = torch.dot(g1, g2).item()
        auto_dot = torch.dot(a1, a2).item()
        assert ghost_dot == pytest.approx(auto_dot, rel=1e-4, abs=1e-5), (
            f"ghost_dot={ghost_dot}, auto_dot={auto_dot}"
        )
        # Norms must also match (same elements, different order)
        assert g1.norm().item() == pytest.approx(a1.norm().item(), rel=1e-4)

    def test_single_conv_ghost_equals_autograd_vector(self):
        """Ghost and autograd inner products match for sum-loss Conv2d."""
        torch.manual_seed(1)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        x1 = torch.randn(1, 3, 6, 6)
        x2 = torch.randn(1, 3, 6, 6)

        model = nn.Sequential(conv)
        model.train()

        def get_ghost(x):
            with MultiLayerBackwardGhostManager([conv], keep_raw=True) as hm:
                model.zero_grad(set_to_none=True)
                y = model(x)
                _sum_loss(y, torch.empty(0)).backward()
                raw_a, raw_e = hm.raw_torch_blocks()
            return _ghost_matrix_torch_from_raw_blocks(raw_a, raw_e)[0]

        def get_auto(x):
            model.zero_grad(set_to_none=True)
            y = model(x)
            _sum_loss(y, torch.empty(0)).backward()
            parts = [conv.weight.grad.reshape(-1)]
            if conv.bias is not None:
                parts.append(conv.bias.grad.reshape(-1))
            return torch.cat(parts).float()

        g1 = get_ghost(x1)
        g2 = get_ghost(x2)
        a1 = get_auto(x1)
        a2 = get_auto(x2)

        ghost_dot = torch.dot(g1, g2).item()
        auto_dot = torch.dot(a1, a2).item()
        assert ghost_dot == pytest.approx(auto_dot, rel=1e-4, abs=1e-5), (
            f"ghost_dot={ghost_dot}, auto_dot={auto_dot}"
        )


class TestLayerGhostDotsSDP:
    """_layer_ghost_dots_from_raw_blocks matches full ghost matrix multiply."""

    def test_sdp_and_materialize_paths_match_mm(self):
        torch.manual_seed(2)
        # L=16, ghost_dim=90*10=900 > 256=L^2 -> SDP branch
        conv = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        x = torch.randn(4, 10, 4, 4)
        model = nn.Sequential(conv)
        model.train()
        with MultiLayerBackwardGhostManager([conv], keep_raw=True) as hm:
            model.zero_grad(set_to_none=True)
            y = model(x)
            _sum_loss(y, torch.empty(0)).backward()
            ra, re = hm.raw_torch_blocks()

        g = _ghost_matrix_torch_from_raw_blocks(ra, re)
        dots_ref = g[0:2] @ g[2:4].T
        dots_sdp = _layer_ghost_dots_from_raw_blocks(
            [ra[0][0:2]], [re[0][0:2]], [ra[0][2:4]], [re[0][2:4]],
        )
        assert torch.allclose(dots_sdp, dots_ref, rtol=1e-4, atol=1e-4)

        # L huge, ghost_dim tiny -> materialize branch
        conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        x2 = torch.randn(4, 1, 64, 64)
        m2 = nn.Sequential(conv2)
        m2.train()
        with MultiLayerBackwardGhostManager([conv2], keep_raw=True) as hm2:
            m2.zero_grad(set_to_none=True)
            y2 = m2(x2)
            _sum_loss(y2, torch.empty(0)).backward()
            ra2, re2 = hm2.raw_torch_blocks()

        g2 = _ghost_matrix_torch_from_raw_blocks(ra2, re2)
        dots_ref2 = g2[0:2] @ g2[2:4].T
        dots_m = _layer_ghost_dots_from_raw_blocks(
            [ra2[0][0:2]], [re2[0][0:2]], [ra2[0][2:4]], [re2[0][2:4]],
        )
        assert torch.allclose(dots_m, dots_ref2, rtol=1e-3, atol=1e-3)


class TestGhostLMismatch:
    def test_mismatched_l_raises(self):
        a = torch.randn(2, 4, 10)
        e = torch.randn(2, 5, 10)
        with pytest.raises(ValueError, match="L mismatch"):
            _ghost_matrix_torch_from_raw_blocks([a], [e])


class TestLinearRawUnchanged2D:
    """Pure Linear path: raw blocks stay 2D; no accidental 3D for MLP."""

    def test_two_linear_raw_are_2d(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(6, 4)
                self.fc2 = nn.Linear(4, 2)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        m = Tiny()
        x = torch.randn(3, 6)
        y = torch.tensor([0, 1, 0], dtype=torch.long)
        layers = [m.fc1, m.fc2]
        with MultiLayerBackwardGhostManager(layers, keep_raw=True) as hm:
            m.zero_grad(set_to_none=True)
            logits = m(x)
            nn.CrossEntropyLoss()(logits, y).backward()
            ra, re = hm.raw_torch_blocks()
        assert all(t.dim() == 2 for t in ra)
        assert all(t.dim() == 2 for t in re)
        g = _ghost_matrix_torch_from_raw_blocks(ra, re)
        # Bias-augmented ghosts: (6+1)*4 + (4+1)*2
        assert g.shape == (3, 7 * 4 + 5 * 2)


class TestMaxSpatialPositionsFallback:
    def test_exceeding_max_uses_mean_pooled_2d_raw(self):
        conv = nn.Conv2d(3, 4, 3, padding=1)
        x = torch.randn(1, 3, 32, 32)
        with MultiLayerBackwardGhostManager(
            [conv],
            keep_raw=True,
            max_spatial_positions=50,
        ) as hm:
            model = nn.Sequential(conv)
            model.train()
            model.zero_grad(set_to_none=True)
            y = model(x)
            _sum_loss(y, torch.empty(0)).backward()
            ra, re = hm.raw_torch_blocks()
        assert ra[0].dim() == 2
        assert re[0].dim() == 2


class TestRunForwardBackwardConvIntegration:
    def test_conv_model_has_3d_raw(self):
        class C(nn.Module):
            def __init__(self):
                super().__init__()
                self.c = nn.Conv2d(3, 4, 3, padding=1)
                self.p = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(4, 2)

            def forward(self, x):
                h = torch.relu(self.c(x))
                h = self.p(h).flatten(1)
                return self.fc(h)

        m = C()
        x = torch.randn(2, 3, 8, 8)
        y = torch.tensor([0, 1], dtype=torch.long)
        raw_a, raw_e = _run_forward_backward(
            m,
            [m.c],
            nn.CrossEntropyLoss(),
            x,
            y,
            "cpu",
            raw=True,
        )
        assert raw_a[0].dim() == 3
        assert raw_e[0].dim() == 3


class TestBatchNormFrozenGrads:
    """BatchNorm in eval (under parent train) makes batched sum-loss grads additive."""

    def test_batched_conv_grad_equals_sum_per_sample_when_bn_eval(self):
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.bn = nn.BatchNorm2d(8)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 2)

            def forward(self, x):
                h = torch.relu(self.bn(self.conv(x)))
                h = self.pool(h).flatten(1)
                return self.fc(h)

        m = Net()
        m.eval()
        with torch.no_grad():
            for _ in range(5):
                m(torch.randn(4, 3, 8, 8))

        x = torch.randn(2, 3, 8, 8)
        y = torch.tensor([0, 1], dtype=torch.long)

        was_training = m.training
        bn_prev = [
            (mod, mod.training)
            for mod in m.modules()
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        ]
        m.train()
        for mod, _ in bn_prev:
            mod.eval()
        try:
            m.zero_grad(set_to_none=True)
            loss_b = torch.nn.functional.cross_entropy(m(x), y, reduction="sum")
            loss_b.backward()
            g_batched = m.conv.weight.grad.clone()

            acc = torch.zeros_like(g_batched)
            for i in range(2):
                m.zero_grad(set_to_none=True)
                loss_i = torch.nn.functional.cross_entropy(
                    m(x[i : i + 1]), y[i : i + 1], reduction="sum",
                )
                loss_i.backward()
                acc += m.conv.weight.grad
            assert torch.allclose(g_batched, acc, rtol=1e-4, atol=1e-5)
        finally:
            for mod, prev in bn_prev:
                mod.train(prev)
            if not was_training:
                m.eval()

    def test_batched_grad_not_additive_when_bn_in_train(self):
        """Same setup with BN in train: conv grad for sum-loss is not sum of singles."""
        torch.manual_seed(2)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.bn = nn.BatchNorm2d(8)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 2)

            def forward(self, x):
                h = torch.relu(self.bn(self.conv(x)))
                h = self.pool(h).flatten(1)
                return self.fc(h)

        m = Net()
        m.eval()
        with torch.no_grad():
            for _ in range(5):
                m(torch.randn(4, 3, 8, 8))

        x = torch.randn(2, 3, 8, 8)
        y = torch.tensor([0, 1], dtype=torch.long)

        m.train()
        m.zero_grad(set_to_none=True)
        loss_b = torch.nn.functional.cross_entropy(m(x), y, reduction="sum")
        loss_b.backward()
        g_batched = m.conv.weight.grad.clone()

        acc = torch.zeros_like(g_batched)
        for i in range(2):
            m.zero_grad(set_to_none=True)
            loss_i = torch.nn.functional.cross_entropy(
                m(x[i : i + 1]), y[i : i + 1], reduction="sum",
            )
            loss_i.backward()
            acc += m.conv.weight.grad

        assert not torch.allclose(g_batched, acc, rtol=1e-3, atol=1e-4)


class TestEmbeddingGhostDots:
    def test_embedding_sdp_matches_materialized(self):
        torch.manual_seed(0)
        emb = nn.Embedding(8, 4)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.e = emb

            def forward(self, t):
                return self.e(t).float().sum(dim=(1, 2))

        net = Net()
        toks = torch.tensor([[1, 2, 3], [0, 1, 2], [3, 3, 1], [2, 0, 5]])
        raw_a, raw_e = _run_forward_backward(
            net,
            [emb],
            lambda pred, _: pred,
            toks,
            torch.zeros(toks.shape[0]),
            "cpu",
            raw=True,
        )
        g = _ghost_matrix_torch_from_raw_blocks(raw_a, raw_e, ghost_layers=[emb])
        ref = g[0:2] @ g[2:4].T
        sdp = _layer_ghost_dots_from_raw_blocks(
            [raw_a[0][0:2]],
            [raw_e[0][0:2]],
            [raw_a[0][2:4]],
            [raw_e[0][2:4]],
            ghost_layers=[emb],
        )
        assert torch.allclose(sdp, ref, rtol=1e-4, atol=1e-4)


class TestLayerNormGhost:
    def test_layernorm_ghost_matches_autograd(self):
        torch.manual_seed(1)
        ln = nn.LayerNorm(6)
        # Single-sample batch so hooked per-sample h_gamma/h_beta match autograd.
        x = torch.randn(1, 4, 6, requires_grad=True)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = ln

            def forward(self, t):
                return self.ln(t).sum()

        net = Net()
        with MultiLayerBackwardGhostManager([ln], keep_raw=True) as hm:
            net.zero_grad(set_to_none=True)
            y = net(x)
            y.backward()
            ra, re = hm.raw_torch_blocks()
        g = _ghost_matrix_torch_from_raw_blocks(ra, re, ghost_layers=[ln])
        net.zero_grad(set_to_none=True)
        net(x).backward()
        auto = torch.cat([ln.weight.grad.flatten(), ln.bias.grad.flatten()]).float()
        assert torch.allclose(g[0], auto, rtol=1e-4, atol=1e-4)


class TestConvTranspose2dGhost:
    def test_conv_transpose_matches_autograd_weight(self):
        torch.manual_seed(3)
        ct = nn.ConvTranspose2d(2, 3, 4, stride=2, padding=1, bias=False)
        x1 = torch.randn(1, 2, 6, 6)
        x2 = torch.randn(1, 2, 6, 6)
        model = nn.Sequential(ct)

        def get_ghost(x):
            with MultiLayerBackwardGhostManager([ct], keep_raw=True) as hm:
                model.zero_grad(set_to_none=True)
                y = model(x)
                _sum_loss(y, torch.empty(0)).backward()
                ra, re = hm.raw_torch_blocks()
            return _ghost_matrix_torch_from_raw_blocks(ra, re)[0]

        def get_auto(x):
            model.zero_grad(set_to_none=True)
            y = model(x)
            _sum_loss(y, torch.empty(0)).backward()
            return ct.weight.grad.reshape(-1).float()

        g1 = get_ghost(x1)
        g2 = get_ghost(x2)
        a1 = get_auto(x1)
        a2 = get_auto(x2)

        # Inner products and norms must match (layout-agnostic)
        ghost_dot = torch.dot(g1, g2).item()
        auto_dot = torch.dot(a1, a2).item()
        assert ghost_dot == pytest.approx(auto_dot, rel=1e-3, abs=1e-4), (
            f"ghost_dot={ghost_dot}, auto_dot={auto_dot}"
        )
        assert g1.norm().item() == pytest.approx(
            a1.norm().item(), rel=1e-3, abs=1e-4,
        )


class TestVAEDeterministicGrad:
    def test_eval_mode_same_grad_twice(self):
        """FashionVAE-style reparameterization is deterministic in eval()."""

        class MiniVAE(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc_fc = nn.Linear(8, 4)
                self.fc_mu = nn.Linear(4, 2)
                self.fc_logvar = nn.Linear(4, 2)
                self.dec = nn.Linear(2, 8)

            def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
                if self.training:
                    std = (0.5 * logvar).exp()
                    eps = torch.randn_like(std)
                    return mu + eps * std
                return mu

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = torch.relu(self.enc_fc(x))
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = self.reparameterize(mu, logvar)
                self._last_mu = mu
                self._last_logvar = logvar
                return self.dec(z)

        torch.manual_seed(0)
        m = MiniVAE()
        m.eval()
        x = torch.randn(1, 8)
        target = x

        def elbo_loss(recon, tg):
            import torch.nn.functional as F

            bce = F.mse_loss(recon, tg, reduction="mean")
            mu = m._last_mu
            logvar = m._last_logvar
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return bce + kl

        grads = []
        for _ in range(3):
            m.zero_grad(set_to_none=True)
            recon = m(x)
            elbo_loss(recon, target).backward()
            grads.append(m.enc_fc.weight.grad.clone())
        assert torch.allclose(grads[0], grads[1], rtol=0, atol=0)
        assert torch.allclose(grads[1], grads[2], rtol=0, atol=0)
