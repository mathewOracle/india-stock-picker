"""Makes `tests` a regular package: pysentiment2 installs a stray top-level `tests` package into site-packages,
and a namespace package (no __init__) always loses to it."""
