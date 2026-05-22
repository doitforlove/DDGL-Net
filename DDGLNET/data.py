import pickle
from typing import Any, Iterator
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler

class DailyBatchSampler(Sampler):

    def __init__(self, data_source: Any, shuffle: bool = False, drop_last: bool = False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.drop_last = drop_last

        index = pd.Series(index=self.data_source.get_index(), dtype="float32")
        self.daily_counts = index.groupby("datetime").size().to_numpy()
        self.daily_starts = np.roll(np.cumsum(self.daily_counts), 1)
        self.daily_starts[0] = 0

    def __iter__(self) -> Iterator[np.ndarray]:
        end = max(0, len(self.daily_counts) - int(self.drop_last))
        day_ids = np.arange(end)
        if self.shuffle:
            np.random.shuffle(day_ids)

        for day_id in day_ids:
            start = self.daily_starts[day_id]
            count = self.daily_counts[day_id]
            yield np.arange(start, start + count)

    def __len__(self) -> int:
        return max(0, len(self.daily_counts) - int(self.drop_last))


def load_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def create_daily_loader(data: Any, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    sampler = DailyBatchSampler(data, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(data, batch_sampler=sampler)
