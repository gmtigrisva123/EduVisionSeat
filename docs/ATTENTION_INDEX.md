# VOTO — Visible On-Task Orientation

The complete definition of the index, together with its formulas, thresholds and
scientific basis.

> **The name is a design decision, not a play on words.** See
> [ETHICS_AND_LEGAL.md](ETHICS_AND_LEGAL.md) §1. In summary: the European
> Commission guidelines (C(2025) 884 final) name the inference of "student
> interest and attention" explicitly as a **prohibited practice**, whereas
> measuring gaze direction and eye movement as a geometric observation is **not
> prohibited**. Fines of up to 7% of worldwide turnover.

---

## 1. Three tiers of head-angle estimation

| Tier | Condition | Method | conf | Available signals |
|---|---|---|---|---|
| **full** | IOD ≥ 40 px | 4×4 matrix from FaceLandmarker | 1.00 | head angles + EAR + MAR + iris |
| **pose_only** | 20 ≤ IOD < 40 px | as above, or 6-point solvePnP | 0.70 | head angles **only** |
| **proxy** | no face available | proxy from pose landmarks | **0.35** | approximate angles only |

The `proxy` tier matters more than one would expect: within a 1080p frame of a
30-seat classroom, students in the back row typically have an IOD < 20 px.
**Every face-based signal therefore fails precisely where it is needed most.**

### 1.1 Why the transformation matrix is better than a hand-written solvePnP

MediaPipe solves PnP internally on its own metric mesh and returns a clean 4×4
rigid matrix (`det(R) = 1.0`). A hand-written 6-point PnP is an
**ill-conditioned** problem for near-coplanar configurations — ±5-10° of noise
superimposed on any systematic offset.

If solvePnP must nevertheless be used: **the detail that almost every online
tutorial gets wrong** is that the canonical face model is defined **y-up,
z-out-of-face**, whereas OpenCV uses **y-down, z-into-scene**. Passing it in
directly makes a front-facing face return R ≈ Rx(180°), with Euler angles of
±180 and inverted signs. `src/pose/pose_landmarks.py` already negates Y and Z (two
sign inversions = a rotation of π about the X axis, still a proper rotation).

In addition: `cv2.RQDecomp3x3` returns **degrees**, not a normalised quantity.
The line `angles[0] * 360` that has propagated across Medium and YouTube posts
is **meaningless** — it only "works" because it is paired with a ±10 threshold
that is every bit as arbitrary.

### 1.2 Pose-based proxy — scale-invariant normalisation

Basis: Araya & Sossa-Rivera (Frontiers in Robotics and AI, 2021) filmed primary
school classrooms from behind — "in most of the scene the students' faces are
not visible" — and used exactly two proxies: the ratio of the horizontal
distance from the nose to the centre of the face, and the degree of
foreshortening of the shoulder segment.

```python
sho_w        = |left_shoulder - right_shoulder|     # primary scale measure
yaw_shoulder = (nose.x - shoulder_mid.x) / sho_w
yaw_ear_geo  = 2·(nose.x - ear_mid.x) / |left_ear - right_ear|
yaw_ear_vis  = (vis_left_ear - vis_right_ear) / (vis_left + vis_right)
roll         = atan2(Δeye.y, Δeye.x)
pitch_proxy  = (nose.y - eye_mid.y) / max(IOD, 0.25·sho_w)
neck_drop    = (eye_mid.y - shoulder_mid.y) / sho_w
```

**Three mandatory rules:**

1. **Always divide by an IN-IMAGE length that scales with the subject**, never
   by the image dimensions. Shoulder width is the best such length (it remains
   visible once the face is lost); IOD is the second best but **collapses to 0
   at oblique angles**, so it is never used on its own.
2. **A per-student baseline beats any global constant.** Report the deviation
   from that student's own baseline, not the raw value. This absorbs the
   geometry of the seat (a student at the left-hand edge of the room
   **legitimately** turns ~25° to see the board), the camera tilt, and
   individual physique.
3. **Map the proxy to degrees by FITTING, not by guessing.** See
   `scripts/fit_proxy_mapping.py`. The `PROXY_*_GAIN_DEG` constants in the code
   are **initial values**.

---

## 2. Eye and mouth indices

### 2.1 EAR — Soukupová & Čech (CVWW 2016)

```
EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)
```

FaceMesh indices:
- Right eye: `[33, 160, 158, 133, 153, 144]`
- Left eye: `[362, 385, 387, 263, 373, 380]`

**Thresholds — this is a question of FAIRNESS, not merely of accuracy.**

The original paper uses a fixed threshold of **0.2**. The widely copied
PyImageSearch implementation uses **0.3**. The gap between 0.2 and 0.3 is real,
and it is itself the reason why **no global constant may be used**.

Baseline EAR varies between individuals from ~0.20 to ~0.40 depending on eyelid
shape, eyelid crease, glasses, and camera angle. **A fixed threshold will
mislabel some students as "drowsy" purely on the basis of facial morphology —
and that morphology correlates with ethnicity. For Vietnamese students this is
not a hypothetical risk.**

→ The system uses **75% of that student's own baseline EAR** (Ersoy et al. 2025
report 93.2% eye-state accuracy with this approach, compared with a fixed
threshold).

The baseline is taken at the **75th percentile**, not the median — the median is
dragged down by blink frames.

**Correct for roll/yaw before comparing against the threshold.** EAR is computed
in the image plane, so roll and yaw contract the horizontal denominator and
**inflate EAR** at oblique angles. The system gates the eye signal off when
`|yaw| > 35°`.

### 2.2 PERCLOS — Wierwille et al., NHTSA DOT HS 808 247 (1994)

Definition: **the proportion of time within the window during which the eyes are
at least 80% closed** (the P80 variant is the standard; P70 and EYEMEAS also
exist).

Window: classically **60 seconds**. Common implementation thresholds: ≥ 0.15 for
a level-1 warning, ≥ 0.30 for level 2, plus an override for continuous eye
closure ≥ 2 seconds.

**Blink rejection is mandatory.** A closure of < 400 ms is a blink; without
rejection, PERCLOS is no more than a noisy proxy for blink rate.

> **A caveat that must be recorded in the product documentation:** the systematic
> review in *SLEEP Advances* 2023 notes that the definition of PERCLOS "varies
> considerably across studies", and records the failures frankly: no effect at
> moderate drowsiness, failure in older drivers, ineffective in aviation tasks,
> and inconsistent results between devices measuring the same quantity. PERCLOS
> is the BEST-validated signal in this group, and it remains this precarious.
>
> **Furthermore: PERCLOS was validated for sleep-deprived drivers, not for bored
> 15-year-old students. No study has validated PERCLOS as a measure of learning
> engagement.** Use it solely as a drowsiness flag.

### 2.3 Blink rate — NOT included in the index

Measurements taken in situ (Chidi-Egboka et al., IOVS 2023):

| Activity | Blink rate |
|---|---|
| Conversation | 32.4 ± 12.4 /min |
| Walking | 31.3 ± 15.5 /min |
| **Reading** (paper, laptop, TV, **or phone**) | **10.7 ± 9.7 /min** |

Two consequences. First, the frequently quoted figure of "15-20/min is normal"
is a poor baseline — the task determines everything. Second, and critically for
this problem: **reading reduces the blink rate to one third, and the effect is
identical for a paper book and for a phone.** Blink rate therefore **cannot
distinguish** a student reading a textbook from a student reading Instagram.

The standard deviation (±9.7 on a mean of 10.7) also means that this measure is
close to useless without individual calibration. The direction of the effect is
still disputed.

→ **Log it; do not score it.**

### 2.4 MAR and yawning

```
MAR = (|p81−p178| + |p311−p402|) / (2·|p78−p308|)     # use the INNER lip contour
```

Commonly encountered fixed threshold: MAR > 0.5-0.6 sustained for ≥ 1.5-2
seconds. The system uses **140% of the individual baseline MAR**.

**A SIMULTANEOUS eyelid droop must be required.** Talking, laughing, drinking,
and answering aloud all raise MAR. In a genuine yawn the eyes close.

---

## 3. Component signals

Every signal lies within [0,1]. **`None` = NOT MEASURABLE, not 0.**

### A. Orientation — weight 0.35

```
o_yaw = clip((Y₀ + Δ − |yaw − yaw_baseline|) / Δ, 0, 1)      Y₀ = 30°, Δ = 15°
```

The ±30° threshold **is documented in the literature** (Xue 2025 uses an outer
band of ±45°; the field converges on ±30°). The **±10°** figure that dominates
blog posts and GitHub repositories (`if y < -10: "Looking Left"`) **has no
empirical basis whatsoever** — it is a display heuristic from a single Medium
article, copied onwards ever since. Do not cite it.

`yaw_baseline` is the viewing direction expected from that seat towards the
board — computed from the chair coordinates, or learned from the modal yaw of
that student during the first 60 seconds. **It is mandatory (see §5).**

### B. Head-down state — NEUTRAL by default

**This is the most important structural decision in the index.**

If the pitch falls within the head-down band (−60°..−20°, that is, the
reading/writing band of Xue 2025):

| Condition | Result |
|---|---|
| A **phone** is detected in the region | `d = 0`, **overrides** VOTO ≤ 0.25 |
| A **writing** signature | `d = 1`, contributes positively |
| Undetermined | **`d = NaN`, removed from the weighted sum, renormalise** |

**Rationale:** a student with their head down taking notes is the **most
focused** student. A naive system that treats head-down as inattention would
**systematically penalise precisely the hardest-working students** — the most
damaging kind of error it is possible to make.

Xue (2025) hard-codes a pitch of −60°..−30° as "reading/writing = focused". That
is the correct intuition.

**How to distinguish them, in order of decreasing reliability:**

1. **Object detection.** `cell phone` is COCO class 67 → all but free. A phone in
   the hands within the head-down region is **decisive evidence**. This is the
   only solution with high accuracy.
2. **Hand and forearm geometry.**
   - *Writing*: one wrist nearly stationary at the desk surface with small
     high-frequency oscillation, the other hand holding the notebook, the
     forearm close to horizontal, **the two wrists far apart** (> 0.6·shoulder
     width).
   - *Phone*: both hands raised and **drawn together towards the centre**, above
     the desk surface, the distance between the wrists **small**, with little
     fine tremor.
3. Nothing else is reliable. The MDPI systematic review (Sensors 2025, 25(2),
   373) surveys the entire field and **puts forward no method** for
   distinguishing note-taking from phone use. SCB-Dataset treats "head down" and
   "reading/writing" as two **separate classes** — a tacit admission that posture
   cannot separate them.

### C. Posture — weight 0.20

```
p = clip((torso_angle − 130°) / 25°, 0, 1)
```

The 155° threshold comes from AIDA-IA (Frontiers in Education 2026). Posture
signals are the **strongest indicators** in Zaletelj's study, AND they **degrade
gracefully at distance** — precisely where face-based signals fail completely.

Because the hips are usually hidden by the desk when seated, the torso angle is
computed from the **shoulder line → eye midpoint**, not from the full spine.

### D. Stability — weight 0.20

```
s = exp(−(σ²_yaw + σ²_pitch) / (2σ₀²))          σ₀ ≈ 12°
```

Over a 45-second window, using the **CIRCULAR standard deviation** (the ordinary
standard deviation is severely wrong when angles cross the ±180° boundary).

Basis: LightNet (Ji et al. 2025) on DAiSEE — low engagement ⟹ faster head
rotation with wider amplitude; high engagement ⟹ appreciably lower
yaw/pitch/roll variance. The Jonckheere-Terpstra test confirms a monotonic
trend. **Yaw is the strongest single indicator, at η² = 0.094** — that is, it
explains ~9% of the variance. The effect is real, and it is **small**.

### E. Alertness — weight 0.15, gated

Only when `IOD ≥ 40 px` **and** `|yaw| < 35°`:

```
a = 1 − clip((PERCLOS_P80 − 0.08) / 0.22, 0, 1)
```

So `a = 1` below a PERCLOS of 0.08, falling to 0 at 0.30, and passing through
0.68 at the conventional warning point of 0.15. `a = 0` is forced when the eyes
remain closed for ≥ 2 seconds.

**If the gate is not satisfied: `a = NaN` and renormalise. Never impute a
value.**

### F. Yawning — weight 0.10, gated

```
y = 1 − clip(yawns_per_5_minutes / 3, 0, 1)
```

---

## 4. Fusion and the validity floor

```
VOTO(t) = Σ_{i ∈ V(t)} wᵢ·xᵢ(t)  /  Σ_{i ∈ V(t)} wᵢ
```

where `V(t)` is the set of **valid** signals at time t.

**Validity floor:** if `Σ_{i∈V} wᵢ < 0.5`, output **`VOTO = null` together with a
reason code** rather than a number.

> A confident-looking score built from 2 of 6 signals is worse than no score at
> all.

**The weights are PRIORS, not scientific findings.** The only defensible weights
are those fitted on ground truth that you have actually collected. State this
explicitly in every report.

---

## 5. Per-student baselines — a FAIRNESS requirement

This is not a performance optimisation. It is a fairness requirement.

Students seated at the left- or right-hand edge of the room **must** turn their
heads ~25-30° in order to see the board. Without individual baselines, these
students are scored low **systematically**. Because seating position often
correlates with academic performance, conduct or eyesight, this creates a
**feedback loop that penalises exactly the group of students who are already
disadvantaged**.

The baseline is locked using the **median** (robust to outliers) after the first
60-90 seconds.

> **An implementation detail worth noting:** when the baseline locks, the yaw
> offset makes a step change. Without clearing the sliding window, the
> **stability** signal registers a spurious jump and falls — penalising the
> student merely because the system has just finished calibrating. The code
> clears the window at the moment of locking; a unit test guards this
> (`test_baseline_lock_does_not_spike_instability`).

---

## 6. Temporal chain

```
30 fps raw
  → per-frame features
  → 1-second median filter                    (Zaletelj: on the features)
  → 1 Hz feature stream
  → 11-second Gaussian smoothing on the FEATURES   (+3.1%)
  → per-second VOTO
  → 45-second sliding window
  → 11-second median filter on the OUTPUT          (+7.8%)
  → Schmitt hysteresis → reported every 5 seconds
```

The key result of Zaletelj & Košir (2017): **smoothing the OUTPUT is more
beneficial than smoothing the INPUT** (+7.8% against +3.1%), because attentional
state is strongly autocorrelated over time. Both are applied.

Windows used in the literature: 13 frames (Soukupová), 11 seconds (Zaletelj), 60
seconds (MATT/PERCLOS), 2 minutes (Farag), 1 fps (AIDA-IA). The field converges
on: 1 Hz features → a 30-60 second window.

---

## 7. Schmitt hysteresis

Never apply a threshold directly to the raw index.

| Band | Enter | Exit |
|---|---|---|
| **low** | VOTO < 0.35 sustained ≥ 20 s | VOTO > 0.45 sustained ≥ 10 s |
| **high** | VOTO > 0.65 sustained ≥ 20 s | VOTO < 0.55 sustained ≥ 10 s |
| **medium** | all remaining cases | |

The hysteresis gap of 0.10 and the **asymmetric** dwell times (slow to enter at
20 s, quick to leave at 10 s) both suppress flicker and **bias the system
towards NOT labelling a student as inattentive**. This bias is deliberate.

The displayed labels are deliberately neutral:

| Band | Label |
|---|---|
| high | "Orientation stable towards the instruction" |
| medium | "Orientation variable" |
| low | "Orientation frequently away from the instruction" |
| unknown | "Insufficient valid signal" |

---

## 8. Class-level output — the recommended default

Recognition at the contextual level attains **κ ≈ 0.60**; at the individual level
only **κ ≤ 0.10**.

Class-level aggregate figures are **at once more scientifically valid, more
useful to the teacher, and considerably less legally risky**.

`class_level_curve()` outputs the mean VOTO in 30-second bins together with a
**95% confidence interval**. Feng et al. (2025) found genuine structure within a
50-minute lesson: a "golden period" at minutes 5-20, with troughs at minutes
20-25 and in the final 5 minutes. That is the kind of information this curve is
able to reveal — and it **requires no individual-level inference whatsoever**.

---

## 9. Empirical basis — read this before trusting any number

| Fact | Source |
|---|---|
| **Teacher annotators disagree with ONE ANOTHER by 41.6° RMSE** on gaze yaw, and 41.1° on body yaw, from classroom video. The same annotator re-scoring the very same image: **37.0°** | Araya & Sossa-Rivera 2021 |
| Best ML model: **35.5° RMSE** — better than humans, but the concept carries ~40° of inherent ambiguity | as above |
| Head yaw explains **η² = 0.094** of the variance in engagement | Ji et al. 2025 |
| Class level κ ≈ 0.60 · individual level κ ≤ 0.10 | synthesis |
| Learners mind-wander **~30% of the time** regardless of the situation | mind-wandering review |
| Gaze from MediaPipe landmarks: error of **7-10°**, with **18-26% of samples unusable** | Agostinelli et al. 2026 |
| DAiSEE is severely imbalanced: **4 samples** at level 0 against ~850 at each of levels 2 and 3 | Goyal et al. 2026 |

> **If four trained teachers cannot agree to within 40° on which way a student's
> head is turned in classroom video, then no ±30° threshold is measuring what it
> claims to be measuring.**

On the imbalance in DAiSEE: a model that always predicts "engaged" will score
very highly. **Any paper quoting top-1 accuracy on DAiSEE without per-class
recall conveys no information at all.**

And one formula to avoid: Sharma et al. (IET CV 2019) define "Concentration
Index = probability of the dominant emotion × emotion weight". **That is
precisely the emotion recognition prohibited in education by Article 5(1)(f). Do
not use that formula.**
