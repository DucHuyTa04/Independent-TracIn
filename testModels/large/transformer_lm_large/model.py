"""Large GPT-style character-level language model (~26M params).

Scales ``TinyGPT`` to production-relevant size:
  - 6 transformer blocks, embed_dim=512, 8 heads, mlp_dim=2048

Ghost coverage (hooking all nn.Linear): ~60% of total params.
Total: ~26M params.
"""

from testModels.medium.transformer_lm.model import TinyGPT


def build_large_gpt(vocab_size: int = 96, ctx_len: int = 128) -> TinyGPT:
    """Return a ~26M param GPT for character-level LM."""
    return TinyGPT(
        vocab_size=vocab_size,
        embed_dim=512,
        n_heads=8,
        n_layers=6,
        ctx_len=ctx_len,
        mlp_dim=2048,
    )
