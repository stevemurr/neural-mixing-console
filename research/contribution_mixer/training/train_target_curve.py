"""Compatibility entry point for historical training commands."""
from nmc.eq.train_target_curve import evaluate, main as train, make_lr_schedule

__all__ = ["evaluate", "make_lr_schedule", "main"]


def main():
    return train(default_cache="dmc-data/v13_precomputed.pt", checkpoint_prefix="v13-target-curve")


if __name__ == "__main__":
    raise SystemExit(main())
