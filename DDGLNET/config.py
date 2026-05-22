from dataclasses import dataclass
import torch

@dataclass
class DDGLConfig:
    """Configuration for DDGL-Net."""

    coarse_feat_dim: int = 158
    fine_feat_dim: int = 158
    market_dim: int = 60

    model_dim: int = 256
    attention_heads: int = 8
    dropout: float = 0.5

    short_window: int = 5
    long_window: int = 30

    num_experts: int = 4
    epochs: int = 50
    learning_rate: float = 1e-5
    grad_clip_value: float = 3.0
    early_stop_loss: float = 0.95
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "ddgl_net.pt"
