# Height estimation from a seated posture

> **The conclusion first, the details afterwards:** feasible at 4-5 cm MAE **if
> and only if** the camera is calibrated AND the arms are visible. If the arms
> are not visible, a *perfect* vision system gives 4.4 cm MAE, whereas looking
> height up by age and sex from the class roster gives 4.8-5.2 cm — the entire
> effort buys 0.5 cm. With uncalibrated video it is **not feasible** (8-15 cm
> MAE, with a large systematic bias).

---

## 1. The anthropometric floor — no pipeline can go below it

The standard error of estimate when predicting height, **assuming perfect
measurement in metres**:

| Model | SEE men | SEE women | SEE pooled |
|---|---|---|---|
| Baseline: guess the mean | 6.86 | 6.42 | 9.00 |
| Sex only | — | — | 6.72 |
| Shoulder width only | 5.81 | 5.29 | 5.72 |
| **Head + shoulders (no arms)** | **5.47** | **4.98** | **5.50** |
| **Head + shoulders + upper arm** | **3.58** | **3.29** | **3.35** |
| Entire upper body | 3.36 | 3.15 | 3.30 |
| Entire upper body + sitting height | 2.25 | 2.11 | — |

Computed from the ANSUR II microdata (n = 4,082 men / 1,986 women).

**The jump from 5.5 down to 3.5 cm comes ENTIRELY from the arm.** If the desk
hides the arms, you are stuck at ~5.5 cm SEE **before any vision error
whatsoever**.

### Comparison against the free baselines

| | MAE |
|---|---|
| Anthropometric floor with arms | ~2.6 cm |
| **Anthropometric floor WITHOUT arms** | **~4.4 cm** |
| **Baseline: table lookup by age × sex** | **4.8-5.2 cm** |
| Baseline: pooled mean, sex unknown | 7.2 cm |

Look closely at the two rows in bold.
`test_arm_free_floor_is_worse_than_roster_baseline` in the test suite encodes
precisely this conclusion.

---

## 2. Choosing the measurement segment

| Rank | Segment | SEE (cm) | k sex-invariant? | Visible when seated? | Conclusion |
|---|---|---|---|---|---|
| **1** | **Upper arm (shoulder→elbow)** | **3.9-4.2** | **Yes (0.1%)** | **Yes** | ★ **USE THIS ONE** |
| 2 | Shoulder→elbow (vertical) | 3.7-4.0 | Almost (0.9%) | Only when the arm hangs straight | Secondary indicator |
| 3 | Sitting height | 4.1-4.3 | Yes (0.5%) | **No** — the seat surface is hidden | Unusable |
| 4 | Forearm | 4.7-4.8 | No (2.9%) | Yes, heavily foreshortened | Weak indicator |
| 5 | Crown→suprasternal notch | 5.2-5.6 | Yes | No landmark available | Discard |
| 6 | **Shoulder width** | **5.3-5.8** | **No (5.5%)** | Yes, but ∝cos(yaw) | Use only as an auxiliary variable |
| 7 | Head / face | 6.0-6.8 | — | Yes, but small | **All but useless** |

### Details of the principal ratios

**Upper arm (acromion–radiale) / height.**
k = 5.239 (men) / 5.233 (women) — **identical to within 0.1%**. This is why it is
the primary indicator: the sex need not be known. r = 0.795 / 0.793.

**Shoulder width (biacromial) / height.**
The ratio is 0.2367 (men) / 0.2244 (women) — **a sex difference of 5.5%**. If the
sex is unknown, this alone injects ±2.8% ≈ ±4.6 cm of systematic error. The ratio
also rises during male puberty (androgen-driven broadening of the shoulders), so
it is **not stable with age** in the 6-18 group. r is only 0.53.

**The Cormic index (sitting height / standing height).**

| Population | Cormic |
|---|---|
| European / Indo-Mediterranean | ~0.520 |
| African | ~0.510 |
| **Asian / East Asian** | **0.530-0.540** |

> **Using the European value of 0.52 for Vietnamese students will OVERESTIMATE
> height by 3-5%, that is, by +5 to +8 cm. This is the largest systematic error
> of the naive approach.**

And the Cormic index **changes with age** (the legs lengthen faster than the
trunk during childhood, it reaches its minimum around the growth peak, and edges
back up at the end of puberty):

| Age | Cormic |
|---|---|
| 6-8.5 | 0.55-0.56 |
| ≥11.5 (girls) | 0.53-0.54 |
| ≥12 (boys) | 0.52-0.53 |

→ **It must be indexed by age × sex × population**, failing which there will be a
systematic error of several centimetres.

**The "7.5 heads" rule.**
This is a rule of **fine art, not a measuring instrument**. ANSUR II: head length
r = 0.37 (men) / 0.36 (women), SEE 6.37/5.99 cm against a height SD of
6.86/6.42 — **a variance reduction of only 13%**. Independent confirmation: a
study of 288 Kosovo-Albanian men regressing height on head height + head
circumference + face length attained R² = 0.262 — **74% of the variance
unexplained**. Bizygomatic width was the worst (r = 0.14).

**Arm span ≈ height (Vitruvius).**
ANSUR II: span/stature = 1.0330 (men) / 1.0195 (women) — the span **exceeds**
height by ~3% in men and ~2% in women, not 1.000. It is not relevant here (a
student seated at a desk never extends to full span), but it is worth recording
so that the reciprocal ratio is not used by mistake — `constants.py` stores
k = 0.9679, not 1.033.

### Systematic bias in the anatomical definitions

**MediaPipe's "shoulder" landmark is the SHOULDER JOINT CENTRE (glenohumeral),
not the ACROMION; the "elbow" is the elbow joint centre, not the radiale point.**
The measured segment is therefore **systematically 2-4 cm shorter than the ANSUR
definition**.

`MEDIAPIPE_TO_ANSUR_CORRECTION` in the code holds an initial estimate of the
correction factor. **The correct procedure is to refit it in situ** — see
`scripts/fit_height_model.py`.

---

## 3. The scale problem

```
s_px = f_px · S_m / Z_m        ⟺        S_m = s_px · Z_m / f_px
```

One equation, two unknowns (Z and f). The ways of pinning down the remaining
unknown:

| Method | ε in scale | Notes |
|---|---|---|
| **ArUco marker** | **~2%** | Best and cheapest. `SOLVEPNP_IPPE_SQUARE` |
| Pre-measured seat depth | ~3.5% | Tape measure, fixed camera. Cheap and durable |
| TCVN reference object | ~5% | A 41 cm chair ±0.5 = 1.2%, but it must be at **the same depth** |
| A person of known height | ~4% | Requires a reference person in the frame |
| Depth Anything V2 / Metric3D | **~20%** | **Far too poor** — see §3.3 |
| `pose_world_landmarks` | — | **Unusable** — see §3.4 |
| Criminisi Single View Metrology | — | Requires the point at which the feet meet the floor — **when a student is seated the feet are hidden by the desk** |

### 3.1 Error propagation

A scale error `ε` → a height error `ε·H`. With H = 165 cm:

| ε | Height error |
|---|---|
| 1% | 1.65 cm |
| **2%** | **3.30 cm** ← level with the anthropometric floor |
| 5% | 8.25 cm |
| 10% | 16.50 cm |

→ **Scale accurate to ~2% is required** if it is not to swamp the anthropometric
floor.

### 3.2 Focal length

```
f_px = (W/2) / tan(HFOV/2)
```

Rule of thumb for webcams and phones (HFOV 50-70°): **0.7·W ≤ f_px ≤ W**. If
calibration returns an f outside this range, the result is **wrong**.

From EXIF: `f_px = f_mm · W_px / sensor_width_mm`.

### 3.3 Depth estimation networks (2024-2026) — why they are NOT used

| Model | Metric? | Error in the field |
|---|---|---|
| Depth Anything V2 | metric variant | MAE 0.454 m, relative error **0.211** |
| Metric3D v2 | yes, zero-shot | 0.867 m / **0.285** |
| Depth Pro (Apple) | yes, absolute | 1.127 m / **0.336** |
| UniDepth v2 | yes, also predicts the intrinsics | δ₁ 95.4 zero-shot indoor |

Even the best model has **~20% relative depth error** in the field. Since the
height error ≈ the depth error (100%), that amounts to **±33 cm for a student of
165 cm**. Even in-domain (NYU indoor), an AbsRel ≈ 0.18 is still 18%.

**They are usable for one purpose only: ordering relative depth**, so as to know
which student is further away and to apply the correct perspective correction —
the absolute scale must still come from an ArUco marker.

### 3.4 `pose_world_landmarks` — why it is NOT used

The Google documentation: *"real-world 3D coordinates in metres, with the origin
at the centre between the hips."*

Empirical measurements of the same person at several image scales:

| Input | Shoulder width | Hip width | Upper arm |
|---|---|---|---|
| crop @1000 px | 33.5 cm | 22.7 cm | 22.3 cm |
| crop @700 px | 33.6 cm | 22.7 cm | 22.2 cm |
| crop @500 px | 34.0 cm | 23.0 cm | 22.2 cm |
| crop @300 px | 33.9 cm | 22.9 cm | 22.0 cm |

The output is **invariant** across a 3.3× range of scales (±0.5 cm). **That very
invariance is the problem:** the metric scale is **not recovered from the image**
at all — it comes from fitting the **GHUM** statistical body model during
training.

The model will return very nearly the same limb lengths for a person of 150 cm
and one of 200 cm, because it is **regressing a canonical body, not measuring**.
With no camera intrinsics as input, no baseline and no scale reference, absolute
metric scale from a single uncalibrated RGB image is a **fundamentally
underdetermined** problem, and MediaPipe resolves that ambiguity by **assuming an
average body**.

Using it to estimate height is **circular reasoning**: you receive back precisely
the mean value you were attempting to improve upon.

**Usable for:** joint angles (scale-invariant — exactly the design intent), left/
right ratios and symmetry, relative variation over time for the same person under
a fixed camera, and features for action classification.

**Not usable for:** anthropometry, clothing sizes, height estimation, or
ergonomic measurement.

---

## 4. Perspective and foreshortening

### 4.1 Landmark noise as a function of distance

`σ_L/L = √2·σ_kp / L_px`. With a realistic σ_kp = 3 px:

| Segment | 3 m | 7 m | 12 m |
|---|---|---|---|
| Shoulder width | 2.6% = 4.3 cm | 6.0% = 9.9 cm | 10.3% = **17.0 cm** |
| Upper arm | 2.9% = 4.8 cm | 6.8% = 11.2 cm | 11.6% = **19.1 cm** |
| Head height | 4.2% = 7.0 cm | 9.8% = 16.2 cm | 16.9% = **27.9 cm** |

**A single frame is hopeless beyond ~5 m.** Averaging N independent frames
reduces the error as √N — 300 frames (10 seconds @ 30 fps) give a reduction
factor of 17×. But the errors are not entirely independent (the pose model has a
systematic per-person bias), so the practical floor is **1-2%**.

Actual sizes in pixels (f = 1371 px, that is, HFOV 70° @ 1920):

| Segment (adolescent) | 3 m | 7 m | 12 m |
|---|---|---|---|
| Sitting height (85 cm) | 388 | 166 | 97 |
| Shoulder width (36 cm) | 165 | 71 | **41** |
| Upper arm (32 cm) | 146 | 63 | 37 |
| Head height (22 cm) | 101 | 43 | **25** |
| Face width (13.5 cm) | 62 | 26 | **15** |

### 4.2 Foreshortening — and the trick that remedies it

A horizontal segment projects as `W·cos(yaw)`:

| yaw | Shortfall | Height bias (H=165) |
|---|---|---|
| 10° | 1.5% | −2.5 cm |
| 15° | 3.4% | −5.6 cm |
| 20° | 6.0% | **−10.0 cm** |
| 30° | 13.4% | **−22.1 cm** |
| 45° | 29.3% | −48.3 cm |

Students in a classroom turn their heads and bodies continually. **This alone
disqualifies shoulder width from the role of primary indicator** with an
uncontrolled camera.

**The temporal percentile trick.** Foreshortening acts in **one direction only** —
it **can only shorten** a segment. Over a long observation window, therefore,
taking the **90-95th percentile** (not the maximum — the maximum picks up
landmark noise) recovers the unforeshortened length **and** removes the bias due
to rotation at the same time.

This is the single most valuable technical trick in the whole pipeline. It is
applicable to the upper arm, the shoulder width and the head width. A unit test
guards it (`test_percentile_aggregation_recovers_unforeshortened_length`).

The same applies to leaning: a student leaning forward by 20° shortens the
crown→shoulder segment by 6% (−10 cm of implied height); 30° gives −22 cm.

### 4.3 Students off the optical axis

A person in image column u is not at distance Z but at the radial distance
`Z·√(1 + ((u−cx)/f)²)`. At the edge of a 70° HFOV frame that factor is **1.19** —
**a 19% error if it is ignored**.

→ Back-projection must go through the **full inverse of K** (`K⁻¹[u,v,1]ᵀ`), not
the abbreviated formula `S = s·Z/f`. `CameraModel.backproject()` does exactly
this; there is a unit test.

### 4.4 Lens distortion

Inexpensive cameras with an HFOV ≥ 90° exhibit 3-8% barrel distortion at the
edges, which converts directly into **5-13 cm of height error**. Always undistort
first, using the coefficients obtained from calibration.

---

## 5. TCVN 7490:2005 — Vietnamese school desks and chairs

The legal constraint: **TCVN 7490:2005**, pursuant to Joint Circular
26/2011/TTLT-BGDĐT-BKHCN-BYT. Tolerance **±0.5 cm**.

| Size | Code | Student height (cm) | Chair (cm) | Desk (cm) |
|---|---|---|---|---|
| I | I/100-109 | 100–109 | 26 | 45 |
| II | II/110-119 | 110–119 | 28 | 48 |
| III | III/120-129 | 120–129 | 30 | 51 |
| IV | IV/130-144 | 130–144 | 34 | 57 |
| V | V/145-159 | 145–159 | 37 | 63 |
| VI | VI/160-175 | 160–175 | **41** | **69** |

> **The commonly used figures of "chair ~45 cm / desk ~75 cm" are WRONG for
> Vietnam.** The tallest chair under the Vietnamese standard is 41 cm, and the
> desk 69 cm. The figures 45/75 come from **European BS EN 1729 size 7** (chair
> 46 / desk 76). Using them by mistake produces a scale error of **+9% ≈ +15 cm
> of height**.

**This table is also a free prior.** If the furniture size in the frame can be
recognised, TCVN immediately yields a height range of **±7 cm** with no vision
whatsoever — equivalent to what the pipeline will produce after all the effort.

> The Vietnamese press (VCCI, Báo Chính phủ) has repeatedly reported that the
> furniture actually in schools **does not comply** with TCVN 7490. **Measure it
> with a tape measure before believing it.**

---

## 6. Vietnamese student height norms

18-year-olds (General Nutrition Survey 2019-2020):

| Year | Men | Women |
|---|---|---|
| 1985 | 159.8 | 150.5 |
| 2010 | 164.4 | 153.4 |
| **2020** | **168.1** | **156.2** |
| 2030 target | 168.5 | 157.5 |

> **Secular trend: +3.7 cm per decade in men.** Every ratio table older than ~10
> years is already obsolete for Vietnam.

Peak growth (SITAR study, n = 15,491 urban Vietnamese students): boys at 12.2
years (9.3 cm/year).

`VN_STATURE_BY_AGE` in `constants.py` holds the complete 6-18 year table together
with the SDs. **This is the baseline that the vision model must beat.**

---

## 7. An honest conclusion on feasibility

### 7.1 Uncalibrated YouTube video — NOT FEASIBLE for absolute figures

There is no metric scale anywhere in the frame. Add to that unknown intrinsics
(±20% on f if the FOV is guessed), unknown yaw, unknown depth, and a typical
resolution that leaves the shoulder width at only 40-90 px:

> **An honest expectation: 8-15 cm MAE, with a large systematic bias.** Worse
> than simply guessing the mean height of Vietnamese students by year group (~5
> cm MAE). **Do not output absolute heights from uncalibrated video.**

What **can** be defended:

- **Relative ranking within a single row of desks** (approximately the same
  depth, so the unknown scale cancels out): Spearman ρ ≈ 0.55-0.70 for a
  mixed-sex class, lower (0.45-0.6) within a single sex and age. Usable for the
  question "who is the tallest in this row", not usable for a figure.
- **Coarse classification into 3 groups** (short/medium/tall relative to the
  class) at 60-70% — only marginally better than chance.
- **Comparison between rows CANNOT be defended** without depth.

### 7.2 A calibrated fixed camera — FEASIBLE at 4-6 cm MAE, conditionally

The conditions: full intrinsic calibration + undistortion; the depth of each seat
measured once with a tape measure; an ArUco marker for verification; **the arms
visible** (hence no desks with a solid front panel); students within ~6 m; ≥ 10
seconds of video per student with 90th-percentile aggregation; sex and age known
from the class roster; and the coefficient k recalibrated on a Vietnamese sample.

> **An honest expectation: 4-5 cm MAE at 3-6 m, degrading to 7-10 cm at 9-12 m.
> Spearman ρ ≈ 0.65-0.75.** The anthropometric floor of 3.3 cm is unattainable in
> practice.

If the arms are **not** visible (a desk with a solid front panel, the common
type): this degrades to **6-8 cm MAE**, worse than the class-roster baseline.
**In that case, do not build a vision system.**

### 7.3 Recommended output

**Relative ranking + grouping by TCVN 7490 size** — which is itself precisely a
height-banding scheme (100-109, 110-119, 120-129, 130-144, 145-159, 160-175).

Reporting **"this student belongs to size V"** is a well-founded conclusion,
conformant with the standard and immediately actionable (furniture of the correct
size can be ordered). Reporting **"this student is 158.3 cm tall"** is not.

And: **always publish the prediction interval (±1.96·SEE, that is, roughly ±9-10
cm) alongside every point estimate. If the confidence interval cannot be stated,
do not display the figure.**

---

## 8. Published work — for calibrating expectations

| Work | Setting | Source of scale | Error |
|---|---|---|---|
| Bieler et al. ICCV 2019 | 12 people, 4-7 m, **jumping on the spot** | gravity (9.81 m/s²), **uncalibrated** | **3.9 cm MAE** |
| Bieler, horizontal jump | | | 6.6 cm |
| Bieler, running | | | 12-19 cm |
| **Population-mean baseline** (their own) | — | — | **8.1 cm MAE** |
| Dey et al. arXiv 1805.10355 | in-the-wild web images | none — pure CNN | **5.56 cm MAE** |
| Li et al. EURASIP 2015 | real CCTV, **standing, walking, full body** | full non-linear regression | 1.39 cm MAE |
| Measurement 2024 | fully calibrated, YOLOv7 | Zhang calibration | 0.02-0.8 cm as published |

**Read these figures critically:**

- The results of 1.39 cm and 0.02-0.8 cm are for people **standing, walking, in
  full body view, fully calibrated**, where the head-to-foot extent is **measured
  directly**. That is an entirely different problem: they **measure** height,
  whereas you must **infer** it from a fragment.
- The figure of 0.02 cm is **not physically credible** — it lies below the
  repeatability of the height measure itself (~0.3 cm), and almost certainly
  reflects an in-sample fit on a curated dataset.
- **The only honest reference point for this problem is 5.56 cm MAE** from
  in-the-wild CNN regression, together with the **8.1 cm population-mean
  baseline**. Note how close those two figures are: the deep learning model beats
  "guess the mean" by only 31%.
- The SMPL/human-mesh literature (SMPLify, HMR2.0, CLIFF) **does not report
  height error separately**, and the scale of the mesh is itself a learned prior —
  so it inherits exactly the circularity problem of §3.4.

---

## 9. Decision checklist

Before trusting any height figure:

- [ ] Is `scale_mode` other than `'none'`? If not → **relative ranking only**
- [ ] Is the camera calibrated (RMS < 0.5 px)? If not → an f error of ±20% ≈ ±33 cm
- [ ] Is `arm_visibility_ratio` ≥ 0.15? If not → **worse than a class-roster lookup**
- [ ] ≥ 200 samples per student? If not → landmark noise is insufficiently averaged
- [ ] Has `local_model` been fitted with **LOO-SEE** (not the in-sample error)?
- [ ] Is LOO-SEE ≤ 5 cm? If not → **STOP, use the class roster**
- [ ] Has the actual furniture been measured, rather than TCVN assumed?
- [ ] Has the Vietnamese Cormic index (0.535) been used, not the European (0.52)?
- [ ] Does the output carry a confidence interval and the TCVN size, not a bare figure?
