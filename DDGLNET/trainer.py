import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .config import DDGLConfig
from .model import DDGLNet


def calc_ic(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float]:
    frame = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if len(frame) < 2:
        return np.nan, np.nan
    return frame["pred"].corr(frame["label"]), frame["pred"].corr(frame["label"], method="spearman")


class DDGLTrainer:
    def __init__(self, config: DDGLConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._set_seed(config.seed)

        self.model = DDGLNet(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.fitted = False

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name)

    @staticmethod
    def _set_seed(seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def _split_batch(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = torch.squeeze(batch, dim=0)
        features = batch[:, :, :-1]
        labels = batch[:, -1, -1]
        return features, labels

    def loss_fn(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    def train_epoch(self, loader: Any) -> float:
        self.model.train()
        losses = []

        for batch in loader:
            features, labels = self._split_batch(batch)
            features = features.float().to(self.device)
            labels = labels.to(self.device)

            pred = self.model(features)
            loss = self.loss_fn(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.grad_clip_value)
            self.optimizer.step()
            losses.append(loss.item())

        return float(np.mean(losses))

    def fit(self, train_loader: Any, valid_loader: Any = None) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        best_ic = -np.inf

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            message = f"epoch={epoch:03d} train_loss={train_loss:.6f}"

            if valid_loader is not None:
                _, metrics = self.predict(valid_loader, load_checkpoint=False)
                valid_ic = metrics["IC"]
                message += (
                    f" valid_ic={metrics['IC']:.4f}"
                    f" valid_icir={metrics['ICIR']:.4f}"
                    f" valid_rankic={metrics['RankIC']:.4f}"
                    f" valid_rankicir={metrics['RankICIR']:.4f}"
                )
                if np.isfinite(valid_ic) and valid_ic > best_ic:
                    best_ic = valid_ic
                    self.save()
            elif train_loss <= self.config.early_stop_loss:
                self.save()
                print(message)
                break

            print(message)

        if not os.path.exists(self.checkpoint_path):
            self.save()
        self.fitted = True

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def load(self) -> None:
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.fitted = True

    def predict(
        self,
        loader: Any,
        dataset_index: Any = None,
        load_checkpoint: bool = True,
    ) -> Tuple[pd.Series, Dict[str, float]]:
        if load_checkpoint and not self.fitted:
            self.load()

        self.model.eval()
        predictions = []
        daily_ic = []
        daily_rank_ic = []

        for batch in loader:
            features, labels = self._split_batch(batch)
            features = features.float().to(self.device)

            with torch.no_grad():
                pred = self.model(features).detach().cpu().numpy().ravel()

            label = labels.detach().cpu().numpy().ravel()
            ic, rank_ic = calc_ic(pred, label)
            daily_ic.append(ic)
            daily_rank_ic.append(rank_ic)
            predictions.append(pred)

        flat_pred = np.concatenate(predictions)
        pred_series = pd.Series(flat_pred, index=dataset_index) if dataset_index is not None else pd.Series(flat_pred)

        ic = np.array(daily_ic, dtype=np.float64)
        rank_ic = np.array(daily_rank_ic, dtype=np.float64)
        metrics = {
            "IC": float(np.nanmean(ic)),
            "ICIR": self._information_ratio(ic),
            "RankIC": float(np.nanmean(rank_ic)),
            "RankICIR": self._information_ratio(rank_ic),
        }
        return pred_series, metrics

    @staticmethod
    def _information_ratio(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) == 0 or np.std(values) == 0:
            return 0.0
        return float(np.mean(values) / np.std(values))
