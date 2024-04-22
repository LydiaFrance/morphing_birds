# Morphing Birds

Initial version.

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Package to plot and animate a hawk shape. Loads a default shape and produces polygons. 


## Installation

```bash
python -m pip install morphing_birds
```

From source:
```bash
git clone https://github.com/LydiaFrance/morphing_birds
cd morphing_birds
python -m pip install .
```

To add to pyproject.toml:
```toml
dependencies = ["morphing_birds @ git+https://github.com/LydiaFrance/morphing_birds"]
```

Then to install the dependencies, navigate to your directory, create and/or activate your Python environment, then install. 

```bash
cd path/to/TargetProject
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Usage

[A basic tutorial of the features are here](https://github.com/LydiaFrance/morphing_birds/blob/main/examples/animate-morphing.ipynb). 

The average shape can be translated, and the body pitch altered with `transform_keypoints`. 

To change the shape of the hawk, use `update_keypoints`. It accepts `[4,3]` or `[8,3]` shape keypoints with the order:

- `wingtip, `primary`, `secondary`, `tailtip` (if points are unilateral, will be mirrored on the left side)
- `left wingtip`, `right wingtip`, `left primary`, ..., `right tailtip` (if points are bilateral)

To animate, it accepts `[n,4,3]` or `[n,8,3]` where n is the number of frames -- same order as before. 

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/LydiaFrance/morphing_birds/workflows/CI/badge.svg
[actions-link]:             https://github.com/LydiaFrance/morphing_birds/actions
[pypi-link]:                https://pypi.org/project/morphing_birds/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/morphing_birds
[pypi-version]:             https://img.shields.io/pypi/v/morphing_birds
<!-- prettier-ignore-end -->
