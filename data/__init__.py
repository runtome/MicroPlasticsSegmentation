from .splits import build_splits

__all__ = ["MicroPlasticsDataset", "build_dataloader", "build_splits"]


def __getattr__(name):
    if name == "MicroPlasticsDataset":
        from .dataset import MicroPlasticsDataset
        return MicroPlasticsDataset
    if name == "build_dataloader":
        from .dataloader import build_dataloader
        return build_dataloader
    raise AttributeError(f"module 'data' has no attribute {name!r}")
