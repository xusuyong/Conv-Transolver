import torch
from torch.utils.data import DataLoader, Dataset


class BaseDataModule:
    @property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(
            self.train_data, collate_fn=collate_fn, **kwargs
        )  # train_loader = datamodule.train_dataloader(batch_size=config.batch_size, shuffle=True)

    def val_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.val_data, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.test_data, collate_fn=collate_fn, **kwargs)
    
    def infer_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.infer_data, collate_fn=collate_fn, **kwargs)
