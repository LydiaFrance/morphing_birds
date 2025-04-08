# Morphing Birds


### SkeletonDefinition

This is like a blueprint for an animal's skeleton. It defines the basic structure that any animal model should have, such as marker names and body sections.

#### HawkSkeletonDefinition

This is a specific blueprint for a hawk's skeleton. It inherits from SkeletonDefinition and adds hawk-specific details. Think of it as a specialized version of the general blueprint, tailored for hawks.

### Animal3D

This is the main class and general-purpose for creating and manipulating 3D animal models. It uses a SkeletonDefinition to know how to structure the animal.

#### Hawk3D

This is a specialised version of Animal3D, specifically for hawks. Hawk3D inherits from Animal3D and uses HawkSkeletonDefinition for hawk-specific details.

### AnimalPlotter
This is a tool for visualising the animal models statically. It works with any Animal3D object (including Hawk3D).

### AnimalAnimate
This is a tool for creating animations of the animal models. It also works with any Animal3D object (including Hawk3D).


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

- `wingtip`, `primary`, `secondary`, `tailtip` (if points are unilateral, will be mirrored on the left side)
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
