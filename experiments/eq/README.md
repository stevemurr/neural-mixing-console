Run from the repository root with the original local cache and best checkpoint.

    uv run python -m v13.reproduce experiments/eq/reproduce-v13.toml --output /tmp/v13-result.json

The recipe pins both input files by SHA-256. It checks the historical song split
and evaluates zero, train-mean, and checkpoint curves over 640 validation windows.
The JSON records per-window squared errors and per-song and overall RMSE in dB.
