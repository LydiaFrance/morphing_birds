# Kestrel3D Marker System Documentation

## Overview
The Kestrel3D class uses a sophisticated marker system to represent and visualize the bird's anatomy. Markers are organized into three distinct categories and can operate in two modes (simple and full).

## Marker Categories

### 1. Ignored Markers
- These markers are completely ignored by the system
- Not loaded from CSV, not stored, and not used in visualization
- Examples:
  - Extra head markers (head_mid, left_head, right_head)
  - Pack markers (backpack and tailpack markers)
- Purpose: Excludes markers that are present in the data but not relevant for analysis

### 2. Fixed Markers
- Loaded from CSV and kept at fixed positions
- Used for visualization but not included in motion analysis
- Stored in `self.fixed_marker_index` and accessed through `self.current_shape`
- Examples:
  - head
  - left_shoulder
  - right_shoulder
- Purpose: Provides structural reference points that don't move relative to each other

### 3. Active Markers
- Main markers used for both visualization and analysis
- Stored in `self.markers` and `self.marker_names`
- Position can be updated and transformed
- Purpose: Represents the moving parts of the bird's anatomy

## Operating Modes

### Simple Mode (`use_simple=True`)
- Used for basic analysis and compatibility with hawk data
- Contains 8 active markers in canonical order:
  1. left_secondprimary_tip, right_secondprimary_tip
  2. left_firstprimary_base, right_firstprimary_base
  3. left_secondary_tip, right_secondary_tip
  4. left_tail_tip, right_tail_tip
- Fixed markers in simple mode:
  - head
  - left_shoulder, right_shoulder
  - left_lastsecondary_tip, right_lastsecondary_tip
- Body sections end with '_simple' suffix in definition but displayed without suffix
- Purpose: Provides compatibility with hawk analysis and simpler visualization

### Full Mode (`use_simple=False`)
- Used for detailed analysis of kestrel-specific features
- Contains 34 active markers following anatomical structure:
  1. Hand wing (primaries) from outermost to innermost
  2. Alula
  3. Mid-wing and wrist
  4. Secondaries
  5. Tail feathers (base to tip, including center)
- Fixed markers in full mode:
  - head
  - left_shoulder, right_shoulder
- More detailed body sections without '_simple' suffix
- Purpose: Provides detailed analysis of kestrel-specific morphology

## Implementation Details

### Marker Loading
```python
# Markers are loaded from CSV with flexible column name handling
x_col = f"{csv_name}_x" if f"{csv_name}_x" in self.data.columns else f"{csv_name}x"
y_col = f"{csv_name}_y" if f"{csv_name}_y" in self.data.columns else f"{csv_name}y"
z_col = f"{csv_name}_z" if f"{csv_name}_z" in self.data.columns else f"{csv_name}z"
```

### Marker Storage
- Active markers: `self._markers` (accessed via property)
- Fixed markers: Part of `self.current_shape` accessed via `fixed_marker_index`
- Marker names: `self.marker_names` for active markers
- Fixed marker names: `self.skeleton_definition.fixed_marker_names(_simple)`

### Polygon Handling
```python
# Polygons store marker names, not indices
self.polygons = {
    'section_name': ['marker1', 'marker2', ...],
    ...
}
```

## Best Practices

1. Always check mode compatibility when adding new features
2. Use `marker_names` to reference active markers
3. Access fixed markers through the skeleton definition
4. Handle both simple and full modes in visualization code
5. Maintain the canonical order of markers in each mode

## Common Pitfalls

1. Mixing simple and full mode markers
2. Forgetting to handle fixed markers in polygon visualization
3. Assuming all markers in CSV should be loaded
4. Not maintaining marker order in transformations
5. Confusing section names with/without '_simple' suffix 