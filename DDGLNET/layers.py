import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dim)
        )

        encoding = torch.zeros(max_len, model_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.size(1)]


class DDGEncoder(nn.Module):

    def __init__(self, model_dim: int, heads: int, dropout: float):
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")

        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.scale = math.sqrt(self.head_dim)

        self.temporal_qkv = nn.ModuleList(
            [nn.Linear(model_dim, model_dim, bias=False) for _ in range(3)]
        )
        self.spatial_qkv = nn.ModuleList(
            [nn.Linear(model_dim, model_dim, bias=False) for _ in range(3)]
        )

        self.norm_temporal = nn.LayerNorm(model_dim)
        self.norm_spatial = nn.LayerNorm(model_dim)
        self.norm_ffn = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
        )

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        score = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weight = self.dropout(torch.softmax(score, dim=-1))
        return torch.matmul(weight, v)

    def _split_heads(self, x: torch.Tensor, batch: int, steps: int) -> torch.Tensor:
        return x.view(batch, steps, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape

        residual = x
        temporal = self.norm_temporal(x)
        q, k, v = [self._split_heads(layer(temporal), batch, steps) for layer in self.temporal_qkv]
        temporal = self._attention(q, k, v).permute(0, 2, 1, 3).contiguous()
        x = residual + temporal.view(batch, steps, dim)

        residual = x
        spatial = self.norm_spatial(x).transpose(0, 1)
        q, k, v = [
            layer(spatial).view(steps, batch, self.heads, self.head_dim).permute(0, 2, 1, 3)
            for layer in self.spatial_qkv
        ]
        spatial = self._attention(q, k, v).permute(0, 2, 1, 3).contiguous()
        x = residual + spatial.view(steps, batch, dim).transpose(0, 1)

        return x + self.ffn(self.norm_ffn(x))


class TemporalAggregation(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(sequence)
        query = hidden[:, -1].unsqueeze(-1)
        weight = torch.softmax(torch.matmul(hidden, query).squeeze(-1), dim=1).unsqueeze(1)
        return torch.matmul(weight, sequence).squeeze(1)


class MarketContextModulator(nn.Module):

    def __init__(self, model_dim: int, market_dim: int, dropout: float):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.weight_generator = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim * 2),
        )
        self.short_norm = nn.LayerNorm(model_dim)
        self.long_norm = nn.LayerNorm(model_dim)
        self.output = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        short_repr: torch.Tensor,
        long_repr: torch.Tensor,
        market_features: torch.Tensor,
    ) -> torch.Tensor:
        short_repr = self.short_norm(short_repr)
        long_repr = self.long_norm(long_repr)

        features = torch.stack([short_repr, long_repr], dim=-1)
        market_state = self.market_encoder(market_features)
        logits = self.weight_generator(market_state).view(short_repr.size(0), short_repr.size(1), 2)
        weights = torch.softmax(logits, dim=-1)
        return self.output(torch.sum(features * weights, dim=-1))


class HybridResidualMoE(nn.Module):

    def __init__(self, model_dim: int, num_experts: int, dropout: float):
        super().__init__()
        self.base_expert = self._expert(model_dim, dropout)
        self.residual_experts = nn.ModuleList(
            [self._expert(model_dim, dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_experts),
            nn.Softmax(dim=-1),
        )

    @staticmethod
    def _expert(model_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_expert(x)
        weights = self.gate(x).unsqueeze(-1)
        residuals = torch.stack([expert(x) for expert in self.residual_experts], dim=1)
        residual = torch.sum(residuals * weights, dim=1)
        return (base + residual).squeeze(-1)
