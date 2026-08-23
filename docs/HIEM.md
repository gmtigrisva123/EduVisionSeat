# HIEM — Height-Invariant Engagement Metrics

> **The conclusion first, the details afterwards:** measuring a behaviour in
> pixels measures three things at once — the behaviour, the student's body size,
> and the distance to the camera. HIEM divides every distance by that student's
> own body scale, which removes the second and third **by construction rather
> than by hoping a model learns to ignore them**. On a controlled sweep the
> resulting features are invariant to 1e-16; with realistic landmark noise the
> Spearman correlation against apparent body size falls from **0.97 to 0.05**;
> and an engagement model trained on HIEM features shows **1/15th** the accuracy
> gap between shorter and taller students that the same model trained on pixels
> does.
>
> What HIEM does **not** do is estimate anyone's height. `S` is a length in
> pixels. See [HEIGHT_ESTIMATION.md](HEIGHT_ESTIMATION.md) §7.1 for why that
> restraint is not modesty but arithmetic.

---

## 1. The problem, stated so it can be tested

A pixel measurement of *"how high is that hand raised"* confounds three things:

| Component | Wanted? |
|---|---|
| The behaviour | **Yes** — this is the signal |
| The student's body size | No — a taller student has longer arms |
| The distance to the camera | No — the back row is imaged smaller |

Train an engagement model on raw pixels and it learns all three. It will then
score the tall student in the front row differently from the short student at
the back who is doing exactly the same thing.

**That error is not random.** Seating position correlates with eyesight, conduct
and attainment, so the mistake lands on the same children every lesson. It is
the same feedback loop that [ATTENTION_INDEX.md](ATTENTION_INDEX.md) §5 makes
per-student baselines mandatory to avoid, arriving through a different door.

The fix is a ratio. Body size and camera distance both act on the image as a
**uniform scaling**, and a ratio of two lengths measured in the same image
region is invariant under scaling. So:

```
HIEM feature = (pixel measurement) / S
```

where `S` is that student's own body scale in pixels. The entire contribution of
this module is making `S` good enough that the division is worth doing.

### 1.1 The guarantee, formally

For any similarity transform `g: x -> s·R·x + t` with `s > 0`:

> `HIEM(g·P) == HIEM(P)` for every feature tagged `Invariance.SIMILARITY`, and
> for every feature tagged `Invariance.GRAVITY` whenever `R` is the identity.

Gravity-referenced features — how high a hand is, how far a head has dropped —
keep the image vertical, because "up" is part of what they mean. They are
invariant to scale and translation but **not** to rotating the camera, and
claiming otherwise would be a lie that no scale test would catch.

Each feature declares its class in `FEATURE_SPECS`, and the test suite iterates
over that registry rather than over a hand-written list. A feature added without
declaring its class fails the suite instead of quietly escaping it.

| Feature | Class | Unit |
|---|---|---|
| `hand_raise` | gravity | body scales |
| `wrist_rise` | gravity | body scales |
| `neck_drop` | gravity | body scales |
| `torso_angle` | gravity | degrees — **bypasses HIEM**, already scale-free |
| `wrist_gap` | similarity | body scales |
| `hand_to_face` | similarity | body scales |
| `neck_axis` | similarity | body scales |
| `head_width` | similarity | body scales |
| `motion` | similarity | body scales per second |
| `motion_articulated` | similarity | body scales per second |

---

## 2. Why `S` is hard, and the four things that fix it

### 2.1 Foreshortening — the percentile trick

Out-of-plane rotation projects a rigid segment **shorter, never longer**. A
student turning to a neighbour has their shoulder width contract by `cos(yaw)` —
13% at 30°. Dividing by that inflates every ratio at exactly the moment the
student moves.

Because the error is one-directional, the upper tail of a long observation
window *is* the unforeshortened length. Aggregate each segment at the **92.5th
percentile**, not the mean and not the maximum:

| Estimator | Recovered length | Bias |
|---|---|---|
| Mean | 91.7 | **−8.3%** |
| **92.5th percentile** | **100.7** | **+0.7%** |
| Maximum | 105.1 | +5.1% |

*(true length 100, 300 samples, yaw uniform on 0–40°, 2% landmark noise —
reproduced by `test_percentile_aggregation_recovers_unforeshortened_length`)*

The maximum is the obvious estimator and the wrong one: it tracks the upper tail
of the **noise** rather than the geometry. 92.5 sits far enough up to have shed
the foreshortening and far enough down to have shed the noise.

This is the trick [HEIGHT_ESTIMATION.md](HEIGHT_ESTIMATION.md) §4.2 calls the
single most valuable one in the pipeline. HIEM applies it per segment, and the
same logic spatially: for a two-sided segment the **longer** side is kept,
because the shorter one is the more rotated one.

### 2.2 Noise in a denominator is a bias, not merely noise

Landmark jitter of ~3 px is 2.6% of a shoulder width at 3 m and 10.3% at 12 m.
Noise in a divisor is worse than noise in a dividend, because `E[x/S] ≠ x/E[S]`:
per-frame normalisation systematically **inflates** every ratio, and it inflates
them most for the back rows, where the relative noise is largest.

The fix is to divide by one locked constant per student instead of a per-frame
value. `BodyScaleEstimator` moves through three declared states:

| State | Condition | What it means |
|---|---|---|
| `instantaneous` | < 8 frames | Whatever this frame says, foreshortening and all. What a single photograph gets. |
| `provisional` | ≥ 8 frames | Foreshortening corrected; the value still moves. |
| `locked` | ≥ 30 frames | A per-student constant. This is the tier the fairness argument needs. |

**The lock is a step change and is announced.** When a denominator changes, every
derived signal jumps, and a variance-based signal reads that jump as instability
— penalising the student for the system having finished calibrating.
`HiemFeatures.scale_lock_event` is true on exactly the one frame of the
transition so downstream windows can be cleared. This is the same failure
[ATTENTION_INDEX.md](ATTENTION_INDEX.md) §5 records for the yaw baseline.

### 2.3 Which segment — fuse them all, by inverse variance

| Segment | Ratio to stature | SEE (cm) | Visibility prior | Fusion weight |
|---|---|---|---|---|
| Upper arm | 0.1910 | 4.05 | 1.0 | **0.424** |
| Shoulder width | 0.2306 | 5.55 | 1.0 | 0.226 |
| Forearm | 0.1460 | 4.75 | 0.6 | 0.185 |
| Head width | 0.0853 | 6.40 | 0.5 | 0.085 |
| Hip width | 0.1670 | 5.90 | 0.4 | 0.080 |

`weight ∝ reliability / SEE²`. Inverse-variance weighting is the right rule for
combining independent estimates of one quantity; the reliability factor then
downweights segments that are anatomically fine but practically unusable when a
desk is in the way. **The reliability column is a PRIOR** — refit it before
quoting it, exactly as ATTENTION_INDEX.md §4 requires of the VOTO weights.

The upper arm dominates because its ratio to stature is **sex-invariant to
0.1%** (k = 5.239 men / 5.233 women). Shoulder width carries a 5.5% sex
difference, which is one reason `S` is never converted to centimetres.

Each segment is divided by its population ratio to give a common
"stature-equivalent in pixels", and the survivors are fused with a **Huber
M-estimator** (c = 1.345) so one bad landmark cannot drag the result.

### 2.4 Outliers — and a measured finding about MediaPipe

Huber alone is not enough, and the reason is worth stating plainly because it
was measured on this repository's own images rather than assumed.

**MediaPipe's visibility score does not flag a badly placed landmark.** Running
the pose model over `data/images/input/classroom*.jpg`:

| Student | Shoulder span | Implied upper arm | Ratio to shoulder | **Elbow visibility** |
|---|---|---|---|---|
| classroom4 #8 | 16.5 px | 100.3 px | **7.55x** | **0.961** |
| classroom1 #2 | 34.0 px | 72.8 px | **2.66x** | **0.956** |
| classroom4 #6 | 31.2 px | 38.0 px | 1.51x | 0.963 |

A desk hides the arms, and MediaPipe answers occlusion by **extrapolating a
plausible limb** rather than by lowering its confidence. A visibility gate — the
obvious defence, and the one already used to decide whether a segment is
measurable at all — catches none of these.

So HIEM adds a **geometric consistency gate**. Two segments of one body, each
divided by its own population ratio, should agree closely; individual variation
in limb proportion is a few percent. Projection widens that, but not without
limit. A segment landing outside **1.5x either way** of the anchor is not an
unusual body, it is a landmark in the wrong place.

**The anchor is shoulder width, not the consensus**, and that choice matters. A
Huber estimator downweights an outlier relative to the consensus — but the upper
arm carries the largest fusion weight, so a confidently-wrong arm *becomes* the
consensus and the correct shoulder measurement is what gets discarded. Anchoring
in advance removes that failure. Shoulder width earns the role because in a
seated classroom it is the only segment whose two endpoints are both large,
high-contrast, above the desk line and essentially never occluded.

Its cost is real and is recorded here rather than hidden: a strongly yawed
shoulder line drags the anchor low. The percentile aggregation removes exactly
that over a window, which is why this gate matters most for single still frames
and least for video.

### 2.5 "Not measured" and "measured but contradictory" are different failures

A student whose arms are behind a desk contributes no arm segments and is
measured from the shoulders alone. That is fine.

A student whose arms *were* measured and contradict the rest of the body is a
detection failure wearing the same low weight. That is not fine — and without
distinguishing the two, classroom4 #8 above publishes a body scale of 76 px for
a student whose true scale is several times that.

`BodyScale` therefore carries both `weight` (survivors) and `rejected_weight`,
and `is_usable` requires the survivors to outweigh the rejects. On the seven
repository images this excludes **8 of 22** detected students — a high rate,
honestly reported, and the right outcome: publishing a score for a pose that
contradicts itself is precisely the failure HIEM exists to prevent.

---

## 3. What the evidence says

All of it is reproduced by `python -m src.hiem.demo`.

### 3.1 Controlled scale sweep — the algebra

One real skeleton from `data/pose/images_man_standing_resized.jpg`, resized
across a 3x range. Behaviour is fixed by construction, so any movement in a HIEM
feature is a defect:

| | Raw pixels | HIEM |
|---|---|---|
| Spread across 0.6x–1.8x | **101.4%** | **≤ 1.3e-15** |

Invariant to machine precision, for all seven distance features.

### 3.2 The same sweep with realistic noise — the engineering

Exact invariance is an algebraic property and survives nothing but algebra. What
matters in the field is whether the dependence survives 3 px of landmark jitter
(240 draws):

| Feature | ρ raw | ρ HIEM | Drop | 95% CI on the drop |
|---|---|---|---|---|
| `hand_raise` | −0.976 | **−0.090** | +0.886 | [+0.754, +0.969] |
| `wrist_rise` | −0.975 | **−0.049** | +0.926 | [+0.792, +0.970] |
| `wrist_gap` | +0.974 | **+0.059** | +0.915 | [+0.786, +0.969] |
| `hand_to_face` | +0.971 | **+0.046** | +0.925 | [+0.786, +0.966] |
| `neck_drop` | −0.967 | **−0.025** | +0.942 | [+0.806, +0.963] |
| `neck_axis` | +0.967 | **+0.018** | +0.949 | [+0.811, +0.962] |
| `head_width` | +0.942 | **+0.024** | +0.919 | [+0.775, +0.939] |

Every interval excludes zero. Largest residual: 0.090.

### 3.3 Rotation control — the taxonomy is honest

A 25° image rotation. Similarity features must not move; gravity features must:

| Feature | Class | Change | Expected |
|---|---|---|---|
| `wrist_gap`, `hand_to_face`, `neck_axis`, `head_width` | similarity | ≤ 3.7e-16 | unchanged ✓ |
| `hand_raise` | gravity | 3.3e-01 | changes ✓ |
| `wrist_rise` | gravity | 1.5e-01 | changes ✓ |
| `neck_drop` | gravity | 7.6e-02 | changes ✓ |

### 3.4 Temporal validation — where per-frame normalisation fails

120 frames from a real skeleton, the student turning ±40° to a neighbour and
back, 3 px noise. Behaviour never changes, so the true hand-raise is a constant
and every deviation is measurement error:

| Estimator | Mean | **Bias** | Std | MAE |
|---|---|---|---|---|
| Per-frame shoulder width | −0.2987 | **−0.0273** | 0.0303 | 0.0328 |
| **HIEM (percentile-locked, fused)** | −0.2629 | **+0.0084** | **0.0043** | **0.0085** |
| Ground truth | −0.2714 | — | — | — |

**3.9x lower MAE, 3.3x lower bias, 7x lower variance.** And the correlation
against apparent scale — which is what a model would learn as a spurious feature:

> per-frame shoulder width **ρ = +0.985** → HIEM **ρ = +0.109**
> (95% CI on the drop: [+0.692, +0.977])

Read the **bias** column, not the standard deviation. Noise averages out over a
window; a bias that tracks head yaw does not, and it is the bias that turns
*"this student turned away"* into *"this student raised their hand higher"*.

### 3.5 Equity audit — the labelled question

400 synthetic students; engagement drives a behaviour ratio identically for
everyone, and apparent body scale is independent of engagement, so a fair model
can gain nothing from it. One model sees the pixel measurement, the other the
same measurement divided by body scale:

| Model input | Overall MAE | Shorter half | Taller half | Gap | Permutation p |
|---|---|---|---|---|---|
| Pixels (no HIEM) | 0.1564 | 0.1737 | 0.1391 | **0.0346** | **0.027 — significant** |
| **HIEM ratios** | **0.0461** | 0.0449 | 0.0472 | **0.0023** | 0.626 — chance |

The pixel model is not merely less accurate; its error is **unevenly
distributed**, at **15.2x** the HIEM model's gap between the two halves of the
class. An overall accuracy figure hides that completely.

It is synthetic, and it has to be: these photographs carry no engagement labels,
and inventing some to produce a chart would be worse than saying so. What is real
is the code path — this is the audit a labelled deployment runs.

### 3.6 Field measurement — the honest, weak result

22 students detected across the seven repository images, spanning a 6.0x range of
apparent body scale; 14 usable after the consistency rule. The invariance audit
on these gives large drops for some features and stubborn residuals for others,
with intervals that mostly straddle zero.

**That result should not be dressed up.** Every scale in a still photograph is
`instantaneous` — one frame affords no percentile and no lock, which is HIEM's
weakest tier by design — and 6 to 14 units is not a sample from which a
correlation can be inferred. A residual is also not automatically a HIEM defect:
taller students may genuinely behave differently, and the two full-body standing
subjects in the set really do hold their arms differently from seated children.

§3.1–§3.4 are the sections where the guarantee is actually demonstrated, because
they are the ones where the behaviour is held fixed by construction.

---

## 4. The audit, and a statistical trap inside it

`src.hiem.fairness` runs two audits that answer different questions.

**The invariance audit needs no labels.** Correlate each feature against body
scale before and after normalisation and watch it collapse. Spearman, not
Pearson: the relationship between a pixel measurement and body scale is monotone
but not linear, and one student sitting unusually close would otherwise dominate
the statistic. Runnable on day one of a deployment, on unannotated data, and
repeatedly thereafter as a regression test.

**The equity audit needs labels.** Split students by body scale and compare
accuracy between the groups.

Two details that decide whether the numbers mean anything:

**Bootstrap over students, not frames.** Frames within a student are strongly
autocorrelated; resampling frames would treat 10 000 correlated observations as
10 000 independent ones and report an interval several times too narrow.

**A gap statistic cannot be significance-tested against zero.** `gap = max − min`
across groups is **non-negative by construction**, so its bootstrap distribution
never straddles zero and a percentile interval excludes zero *always* — including
for a provably fair model. An audit built that way reports discrimination every
time it runs and is worthless in both directions: it cannot exonerate, and its
accusations carry no information. `equity_audit` therefore reports a
**permutation p-value** — shuffle the group labels, keep the group sizes, and
measure how often chance reaches the observed gap.

The demo shows the trap live: the fair model's gap CI is `[+0.0002, +0.0119]`,
excluding zero, at `p = 0.626`.

---

## 5. What HIEM does not do

* **It does not estimate height.** `S` is a pixel length. The stature ratios
  only put segments on a common footing. Converting `S` to centimetres needs a
  calibrated camera and a measured seat depth — `with_metric_scale` is the only
  route, and it is the only thing that sets
  `BodyScale.comparable_across_students`.
* **It does not rank students by size.** A bigger `S` may mean a taller student
  or a nearer one, and an uncalibrated image cannot tell them apart. This is
  enforced in code, not merely documented.
* **It does not absorb the seat geometry.** A student at the edge of the room
  legitimately turns 25–30° to see the board. That needs the per-student yaw
  baseline of [ATTENTION_INDEX.md](ATTENTION_INDEX.md) §5. **HIEM handles body
  size; the baseline handles seat position. Both are required and neither
  substitutes for the other.**
* **It does not touch angles.** Yaw, pitch, roll and the torso angle are already
  dimensionless and pass through untouched, with a test asserting it.
* **It does not fix a bad pose model.** It detects some of the damage — see §2.4
  — and refuses to publish a scale built on a self-contradictory pose. It cannot
  put the elbow back where it belongs.

---

## 6. Integration

HIEM adds no dependency beyond numpy and **changes no existing file**. Two entry
points, depending on what the caller holds:

```python
from src.hiem import HiemTracker, PoseObservation

tracker = HiemTracker()
features = tracker.observe(PoseObservation(
    track_id=7, points=landmarks_px, visibility=vis, timestamp_s=t,
))
features.hand_raise          # body scales, not pixels
features.scale.describe()    # S=728.7px[locked] n=120 w=1.00 disp=0.040 conf=0.78
```

For a caller that already has `PersonFeatures` from `src.pose.pose_landmarks`,
`HiemNormaliser.observe_person_features` restores the pixels from the reported
shoulder-width ratios and re-divides by the locked scale. That upgrades
`wrist_gap`, `wrist_rise` and `neck_drop` from a noisy per-frame divisor to a
stable per-student constant **without one line of `pose_landmarks.py`
changing**. Only the subset that class carries can be recovered; movement and
hand-raise need the full landmark dictionary.

For offline work, `normalise_sequence(observations)` runs two passes so every
frame divides by the same final scale. Online, the first frames of a track can
only use an `instantaneous` scale, so early features are systematically noisier
than late ones — an artefact a model trained on the output will happily learn.
Pass `two_pass=False` to reproduce exactly what a live run would have produced,
which is what an honest deployment evaluation needs.

**Packaging:** `pyproject.toml` lists its packages explicitly rather than letting
setuptools discover them, so `src.hiem` is named there alongside `src.detect` and
`src.pose`. A `pip install -e .` ships all seven modules; tests and `python -m`
runs work either way, via `pythonpath`.

---

## 7. Decision checklist

Before trusting a HIEM number:

- [ ] Is `scale.state` `locked`? If not → the denominator is still moving
- [ ] Is `scale.is_usable` true? If not → **do not publish a score for this student**
- [ ] Is `rejected_weight` ≤ `weight`? If not → the pose contradicts itself
- [ ] Is `scale.dispersion` ≤ 0.18? If not → the student is turning too much for this window
- [ ] Is `scale.confidence()` reported alongside the feature, not dropped?
- [ ] Are the thresholds (`HAND_RAISE_ENTER`, `WRIST_GAP_WRITING`) refitted, or still the shipped priors?
- [ ] Has `SEGMENT_RELIABILITY` been refitted on your own footage?
- [ ] Does the invariance audit run on every release, not once at the start?
- [ ] Is the equity verdict taken from the permutation p-value, **not** from the gap interval?
- [ ] Is `comparable_across_students` true before any ranking by size is reported?
