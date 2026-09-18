# Contributing

Thanks for your interest in this project.

While the accompanying paper is under review, the proposed model architecture
is intentionally withheld from this repository (see the README's "Proposed
architecture release status" section), so pull requests touching model
architecture files won't be accepted until after publication.

Contributions to the surrounding pipeline are welcome in the meantime:

1. Fork the repo and create a feature branch.
2. Keep changes focused and add/update tests in `tests/` where relevant.
3. Run `pytest tests/ -v` before opening a PR.
4. Open a PR describing the change and its motivation.

For bug reports, please open an issue with steps to reproduce, your
environment (OS, Python, PyTorch, CUDA versions), and the full error output.

This project vendors [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
under `ultralytics/`; for issues with the underlying framework itself,
consider checking the upstream repository as well.
