import torch
import torch.nn as nn

from .config import DDGLConfig
from .layers import (
    DDGEncoder,
    HybridResidualMoE,
    MarketContextModulator,
    SinusoidalPositionEncoding,
    TemporalAggregation,
)


class DDGLNet(nn.Module):

    def __init__(self, config: DDGLConfig):
        super().__init__()
        self.config = config
        self.total_stock_dim = config.coarse_feat_dim + config.fine_feat_dim

        self.coarse_projection = nn.Linear(config.coarse_feat_dim, config.model_dim)
        self.fine_projection = nn.Linear(config.fine_feat_dim, config.model_dim)

        self.short_position = SinusoidalPositionEncoding(config.model_dim, config.short_window)
        self.long_position = SinusoidalPositionEncoding(config.model_dim, config.long_window)

        self.short_encoder = DDGEncoder(config.model_dim, config.attention_heads, config.dropout)
        self.long_encoder = DDGEncoder(config.model_dim, config.attention_heads, config.dropout)
        self.short_pool = TemporalAggregation(config.model_dim)
        self.long_pool = TemporalAggregation(config.model_dim)

        self.modulator = MarketContextModulator(
            model_dim=config.model_dim,
            market_dim=config.market_dim,
            dropout=config.dropout,
        )
        self.predictor = HybridResidualMoE(
            model_dim=config.model_dim,
            num_experts=config.num_experts,
            dropout=config.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = x[:, :, : self.config.coarse_feat_dim]
        fine = x[:, :, self.config.coarse_feat_dim : self.total_stock_dim]
        market = x[:, -1, self.total_stock_dim :]

        short = self.fine_projection(fine)[:, -self.config.short_window :]
        short = self.short_pool(self.short_encoder(self.short_position(short)))

        long = self.coarse_projection(coarse)[:, -self.config.long_window :]
        long = self.long_pool(self.long_encoder(self.long_position(long)))

        fused = self.modulator(short, long, market)
        return self.predictor(fused)
