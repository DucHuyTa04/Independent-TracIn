"""Variational Autoencoder for Fashion-MNIST (~300K params)."""

import torch
import torch.nn as nn


class FashionVAE(nn.Module):
    """FC-based VAE: Encoder 784→256→128→(μ,logvar)(32), Decoder 32→128→256→784.

    ``forward()`` returns ``(recon, mu, logvar)`` — all three needed for ELBO loss.

    Ghost layers (decoder): ``dec_fc1``, ``dec_fc2``, ``dec_out``.
    Total: ~305K params.  Ghost coverage ≈ 68%.
    """

    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc_fc1 = nn.Linear(784, 256)
        self.enc_fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_out = nn.Linear(256, 784)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor):
        h = self.relu(self.enc_fc1(x))
        h = self.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.dec_fc1(z))
        h = self.relu(self.dec_fc2(h))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstruction (B, 784) only — mu/logvar captured as buffers.

        The ELBO loss wrapper accesses ``self._last_mu`` and ``self._last_logvar``
        to compute KL divergence.  This keeps the model–loss interface compatible
        with the ghost TracIn pipeline (which expects ``loss_fn(model_output, target)``).
        """
        flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(flat)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        self._last_mu = mu
        self._last_logvar = logvar
        return recon
