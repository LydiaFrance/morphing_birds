# Morphing Birds

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19917701.svg)](https://doi.org/10.5281/zenodo.19917701)

A toolkit for plotting and animating morphing bird (and other animal) wing and
tail shapes in 3D flight data.

## What's new in v0.2.0

Version 0.2.0 is a major restructure. The old per-species class hierarchy
(`Hawk3D`, `Kestrel3D`, `Pigeon3D`, `Spider3D`, `ArbitraryBird3D`) and
hard-coded Python skeleton definitions have been replaced with a single
**config-driven** architecture.

### Key changes

- **One class for all animals.** `Animal3D` now takes a config name or a
  `SkeletonDefinition` — no more species subclasses.
- **YAML configs instead of Python files.** Skeleton definitions (markers, body
  sections, laterality, variants) live in declarative YAML files under
  `morphing_birds/configs/`.
- **Cleaner module layout.** Transforms, scaling, bilateral symmetry, and data
  loading are now separate, importable modules.

### Migration from v0.1.x

```python
# Old
from morphing_birds import Hawk3D
hawk = Hawk3D("data/mean_hawk_shape.csv")

# New
from morphing_birds import Animal3D
hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
```

The existing methods — `update_keypoints`, `transform_keypoints`,
`restore_keypoints_to_average` — still work as before. No changes needed for
downstream plotting or animation code.

## Architecture

### SkeletonDefinition

A pure data container loaded from a YAML config. Defines markers, body sections,
laterality, display names, section styles, validation rules, and named variants.
No subclassing needed — different animals are different configs.

```python
from morphing_birds import SkeletonDefinition

# From a builtin
skel = SkeletonDefinition.from_builtin("hawk")

# From your own YAML file
skel = SkeletonDefinition.from_yaml("path/to/my_animal.yaml")
```

Builtin configs: `hawk`, `pigeon`, `kestrel`, `spider`.

### Animal3D

The main class for creating and manipulating 3D animal models. Pass a builtin
name or a `SkeletonDefinition`, plus your data.

```python
from morphing_birds import Animal3D

# Builtin name + CSV path
hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")

# With a named variant (changes which markers are analysed)
hawk_simple = Animal3D("hawk", data="data/mean_hawk_shape.csv", variant="simple")

# From a numpy array
hawk = Animal3D("hawk", data=my_array)  # shape (n_markers, 3) or (1, n_markers, 3)

# From a DataFrame or dict
hawk = Animal3D("hawk", data=my_dataframe)
hawk = Animal3D("hawk", data={"left_wingtip": [x, y, z], ...})
```

#### Updating and transforming shapes

```python
# Update analysis markers with new keypoint positions
hawk.update_keypoints(new_keypoints)  # (n_right, 3) unilateral or (n_markers, 3) bilateral

# Apply body pitch, yaw, roll, and translation
hawk.transform_keypoints(bodypitch=10, horzDist=0.5)

# Reset to the original loaded shape
hawk.restore_keypoints_to_average()
```

#### Loading motion data

```python
# Load multi-frame data from CSV
motion_data = hawk.load_motion_data("data/hawk_flight.csv")

# Step through frames
hawk.update_to_frame(motion_data, frame_idx=42)

# Remove NaN frames
clean_data, valid_mask = hawk.remove_nan_frames(motion_data)
```

#### Scaling

```python
# Unit conversion
hawk.set_scale(unit_from="mm", unit_to="m")

# Normalise by wingspan or body length
hawk.set_scale(normalise_by="wingspan")

# Direct scaling factor
hawk.set_scale(factor=0.001)
```

#### Marker access

```python
hawk.analysis_marker_names   # names of non-excluded markers
hawk.markers                 # analysis marker coordinates (1, n_analysis, 3)
hawk.right_markers           # right-side analysis markers
hawk.left_markers            # left-side analysis markers
hawk.fixed_markers           # display-only marker coordinates

# Runtime exclusion
hawk.exclude_markers(["left_shoulder", "right_shoulder"])
hawk.include_markers(["left_shoulder", "right_shoulder"])
```

### Plotting

All plotting functions accept any `Animal3D` instance.

```python
from morphing_birds import plot, plot_plotly, animate, animate_plotly

# Matplotlib static plot
plot(hawk, colour="blue", alpha=0.5)

# Plotly static plot
fig = plot_plotly(hawk, colour="lightblue")

# Matplotlib animation
anim = animate(hawk, keypoints_frames)

# Plotly animation
fig = animate_plotly(hawk, keypoints_frames, score_vals=scores)

# Compare multiple shapes
from morphing_birds import plot_plotly_compare, animate_plotly_compare
fig = plot_plotly_compare([hawk1, hawk2], colours=["blue", "red"])
fig = animate_plotly_compare(hawk, [frames_a, frames_b])

# Save animation
from morphing_birds import save_plotly_animation
save_plotly_animation(fig, "output.gif", format="gif", fps=10)
save_plotly_animation(fig, "output.html", format="html")
```

### Writing your own config

Create a YAML file following this structure:

```yaml
name: my_animal
laterality: prefix # "prefix" for left_/right_ naming, or dict for suffix-based

markers:
  - left_wingtip
  - right_wingtip
  - hood
  # ... all markers in order

analysis_exclude:
  - hood # markers not used in shape analysis

body_sections:
  left_wing: [left_wingtip, left_primary, left_secondary]
  right_wing: [right_wingtip, right_primary, right_secondary]
  # ... polygons for visualisation

column_mapping: # optional: map CSV column names to marker names
  csv_column_name: marker_name

display_names: # optional: human-readable names for plots
  left_wingtip: L Wingtip

section_styles: # optional: default colours/alpha per section
  default: { alpha: 0.3 }
  left_wing: { colour: blue }

validation_rules: # optional: spatial sanity checks
  - "left_wingtip.x < right_wingtip.x"

variants: # optional: named subsets
  simple:
    analysis_exclude: [hood, tailpack, left_shoulder, right_shoulder]
```

Then load it:

```python
skel = SkeletonDefinition.from_yaml("my_animal.yaml")
animal = Animal3D(skel, data="data/my_animal.csv")
```

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

Then to install the dependencies, navigate to your directory, create and/or
activate your Python environment, then install.

```bash
cd path/to/TargetProject
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Examples

See the [examples/](examples/) directory for Jupyter notebooks demonstrating
each builtin animal:

- `showcase_hawk.ipynb`
- `showcase_kestrel.ipynb`
- `showcase_pigeon.ipynb`
- `showcase_spider.ipynb`

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
