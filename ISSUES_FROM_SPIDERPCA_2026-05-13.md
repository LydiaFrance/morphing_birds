# Issues discovered while migrating SpiderPCA to morphing_birds v0.2.0

**Date:** 2026-05-13
**Discovered by:** Lydia France (during SpiderPCA Animal3D migration, branch `migrate/animal3d-2026-05-13` on `LydiaFrance/spiderpca`)
**Tested against:** morphing_birds v0.2.0 (tag, commit `a2d0456`)

Two confirmed issues plus one open question.

---

## Issue 1 — `spider3d.markers` mutability footgun (high priority)

### Symptom

In a downstream loop that computes PCA reconstructions and saves animations per principal component, motion drifts progressively across PCs. PC1 is correct; PC2..PCn are wrong and the error compounds. In SpiderPCA's `nb02 full_shape_PCA`, this showed up as leg 3 (and other back legs) appearing to move in skewed directions or with exaggerated extent compared to pre-migration runs.

### Root cause

Two design choices interact:

1. **`Animal3D.markers` is a property over `current_shape`:**

   ```python
   # animal.py L310
   @property
   def markers(self) -> np.ndarray:
       """Analysis marker coordinates from ``current_shape``."""
       return self.current_shape[:, self.analysis_indices, :]
   ```

   So `spider.markers` reflects whatever the instance currently holds — not the originally loaded mean shape.

2. **`animate()` calls `restore_default()` only once, at closure-build time:**

   ```python
   # plotting/matplotlib_animate.py L58
   animal3d_instance.restore_default()
   return FuncAnimation(fig, update_animated_plot, frames=num_frames, ...)
   ```

   The actual frame rendering happens later, when `ani.save(...)` runs the `update_animated_plot` closure for each frame. That closure does call `reset_transformation()` per frame, but `update_keypoints(frames[frame])` writes to both `current_shape` and `untransformed_shape`. After `ani.save()` completes, `current_shape` is the final frame, not the mean — and `spider.markers` returns that.

### The natural loop pattern that breaks

```python
for ii in range(10):
    component_number = [ii]
    reconstructed_frames = reconstruct(
        score_frames, principal_components, spider3d.markers, component_number
    )
    ani = animate(spider3d, reconstructed_frames, ...)
    ani.save(f"PC0{ii + 1}.gif", writer="Pillow", fps=20, dpi=150)
```

After iteration 0, `spider3d.markers` no longer equals the mean shape — it equals the last animation frame of PC1. Iteration 1 then uses that drifted shape as `mu` in `reconstruct(...)`. Each subsequent PC drifts further.

### Verified empirically

```text
BEFORE any animation, mu sum: 0.06799973885542723
AFTER animate() (closure built, not run yet): 0.06799973885542723
AFTER ani.save() (frames rendered): 0.0582222205660097
mu drift max abs: 0.007887624440940435
```

### Downstream workaround (already applied in SpiderPCA)

Manually reset before reading `.markers` inside the loop:

```python
for ii in range(10):
    spider3d.reset_transformation()
    spider3d.restore_default()
    ...
```

Pre-v0.2.0 SpiderPCA notebooks already had this — they were removed during migration on the assumption that `animate()`'s internal `restore_default()` made them redundant. It does not.

### Suggested upstream fix (pick one)

In order of cheapest → most idiomatic:

1. **README note + docstring on `Animal3D.markers`.** Document that `.markers` is mutable and reflects the last update. Recommend `restore_default()` before reading after any animation save.
2. **Add a `default_markers` (or `mean_markers`) accessor** that always returns the originally loaded shape regardless of current state. Most idiomatic — downstream code reads `spider.default_markers` for PCA `mu` and never worries.
3. **Snapshot-and-restore inside `animate()`.** Wrap the returned `FuncAnimation` so that its `save()` method snapshots `current_shape`/`untransformed_shape` before iterating and restores after. Fully transparent fix; some surprise factor if anyone was relying on the post-save state.

Option 2 + 1 together would be ideal.

---

## Issue 2 — `examples/showcase_spider.ipynb` markdown contradicts `configs/spider.yaml` laterality

### Symptom

Markdown narrative in `showcase_spider.ipynb` cell 12:

> Exclude legs 3 and 6 (a right-left pair)

But `configs/spider.yaml`:

```yaml
laterality:
  right: ["1", "2", "3", "4"]
  left: ["5", "6", "7", "8"]
```

Combined with the default `make_marker_pairs` rule (leg N right ↔ leg N+4 left), `Skeleton.get_marker_pairs()` produces:

```text
claw1  <->  claw5
claw2  <->  claw6
claw3  <->  claw7   <- right/left pair of leg 3 is leg 7, NOT leg 6
claw4  <->  claw8
```

So leg 3's right-left pair is leg 7, not leg 6. The markdown comment is wrong relative to the code.

### Suggested fix

Either:
- **Fix the markdown** to say "Exclude legs 3 and 7 (a right-left pair)" (and update the polygon/colour choices in the cell accordingly), or
- **Change the laterality config** if the intended convention is mirror-by-symmetry (front-back mirror), in which case pairs would be 1↔8, 2↔7, 3↔6, 4↔5. But that contradicts the current `right: [1, 2, 3, 4]` declaration, which is anatomically right→left numbered.

Most likely the markdown is the bug — the spider's anatomical right legs are 1–4 (front-to-back) and left legs are 5–8 (front-to-back), so leg 3 (right, mid-back) pairs with leg 7 (left, mid-back). Worth confirming with the original anatomical labelling.

---

## Open question — `animate()` axis-limit scaling change

In `plotting/matplotlib_animate.py`, the axis-limit constant changed between versions:

| version | line | code |
|---|---|---|
| v0.1.0 (refactor branch) | 56 | `lims = keypoints_frames.max() * 1.2` |
| v0.2.0 | 38 | `lims = keypoints_frames.max() * 0.5` |

That's a 2.4× tighter axis box in v0.2.0 → the same animation appears 2.4× larger on screen. Probably intentional (the default zoom level), but worth confirming:

- If intentional, no action — but consider exposing an `axis_scale` (or `lims_scale`) kwarg so downstream can pin to the old behaviour without monkey-patching.
- If unintentional, restore `* 1.2` (or any sensible default like `* 1.0`).

Same observation applies to `animate_compare()`.

---

## Context

These were uncovered during the migration of SpiderPCA from the old `Spider3D` class to `Animal3D('spider', ...)` (`v0.2.0`). PCA pipeline data is byte-identical between v0.1.0 and v0.2.0; the visible difference in animated gifs came entirely from Issue 1 (real bug, downstream-fixable) plus the upstream cosmetic deltas in the open question (`axis_scale`, plus a new `section_styles` block in `spider.yaml` that lowered body-section alpha to 0.2 — also probably intentional, mentioned here only for completeness).

Migration record (downstream): `LydiaFrance/spiderpca` branch `migrate/animal3d-2026-05-13`, fix in commit `20d2041`.
