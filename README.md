# DDGL-Net

This repository contains a minimal PyTorch implementation of **DDGL-Net**
(*Disentangled Dual-Granularity Learning Network*) for market-adaptive stock
return forecasting.

The code keeps only the core model and training pipeline:

- **DDGE**: parallel short-term fine-grained and long-term coarse-grained encoders.
- **MACM**: market-context-conditioned fusion of the two granularities.
- **HR-MoE**: shared base expert plus adaptive residual experts.

## Notes

This release focuses on the core neural architecture and IC/RankIC evaluation.
Portfolio metrics such as annualized return, information ratio, and maximum
drawdown are intentionally omitted because they usually depend on a complete
backtesting stack such as Qlib.
