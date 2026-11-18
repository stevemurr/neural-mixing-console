"""Compatibility imports for historical pan and EQ trainers."""
from nmc.data.splits import session_split
from nmc.eq.dataset import TargetCurveDataset as V13Dataset, collate_examples as collate_v13

__all__ = ["V13Dataset", "collate_v13", "session_split"]
