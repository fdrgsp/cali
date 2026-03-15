# Multi-Well Aggregation — Step-by-Step Walkthrough

This document walks through every aggregation that happens in the **Multi Well** tab,
using small concrete numbers so every formula can be verified by hand.

---

## Setup: the example experiment

We have **two conditions** (`ctrl` and `drug`), each with **two FOVs**, and
**3–4 active ROIs per FOV**.

### Metric 1 — Calcium Peak Frequency (Hz)
*(a per-ROI scalar → uses `_aggregate_fov_data_to_condition_stats`)*

```
Condition "ctrl"
│
├── FOV-A  (4 ROIs)
│     ROI-1 = 0.10 Hz
│     ROI-2 = 0.14 Hz
│     ROI-3 = 0.12 Hz
│     ROI-4 = 0.08 Hz
│
└── FOV-B  (3 ROIs)
      ROI-5 = 0.20 Hz
      ROI-6 = 0.22 Hz
      ROI-7 = 0.18 Hz

Condition "drug"
│
├── FOV-C  (3 ROIs)
│     ROI-8  = 0.30 Hz
│     ROI-9  = 0.35 Hz
│     ROI-10 = 0.28 Hz
│
└── FOV-D  (4 ROIs)
      ROI-11 = 0.40 Hz
      ROI-12 = 0.42 Hz
      ROI-13 = 0.38 Hz
      ROI-14 = 0.39 Hz
```

---

## Step 1 — ROI → FOV

For each FOV compute the mean of its ROI values:

$$
\mu_\text{FOV} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

### FOV-A  (n = 4, values = [0.10, 0.14, 0.12, 0.08])

$$
\mu_A = \frac{0.10 + 0.14 + 0.12 + 0.08}{4} = \frac{0.44}{4} = \mathbf{0.110\ \text{Hz}}
$$

### FOV-B  (n = 3, values = [0.20, 0.22, 0.18])

$$
\mu_B = \frac{0.20 + 0.22 + 0.18}{3} = \frac{0.60}{3} = \mathbf{0.200\ \text{Hz}}
$$

### FOV-C  (n = 3, values = [0.30, 0.35, 0.28])

$$
\mu_C = \frac{0.30 + 0.35 + 0.28}{3} = \frac{0.93}{3} = \mathbf{0.310\ \text{Hz}}
$$

### FOV-D  (n = 4, values = [0.40, 0.42, 0.38, 0.39])

$$
\mu_D = \frac{0.40 + 0.42 + 0.38 + 0.39}{4} = \frac{1.59}{4} = \mathbf{0.3975\ \text{Hz}}
$$

**Step 1 summary table**

| FOV | n | μ (Hz) |
|-----|---|--------|
| A   | 4 | 0.1100 |
| B   | 3 | 0.2000 |
| C   | 3 | 0.3100 |
| D   | 4 | 0.3975 |

---

## Step 2 — FOV → Condition

Each FOV is treated as one independent biological replicate. The condition
mean is the unweighted mean of FOV means and the SEM is computed across
FOV means.

$$
\bar{x}_\text{cond} = \frac{1}{M}\sum_{i=1}^{M} \mu_i
\qquad
\text{SEM}_\text{cond} = \frac{\sigma(\mu_1, \dots, \mu_M)}{\sqrt{M}}
\quad\text{where}\quad
\sigma = \text{std}(\text{ddof}=1)
$$

### Condition "ctrl"  (FOV-A + FOV-B, M = 2)

$$
\bar{x}_\text{ctrl}
= \frac{0.1100 + 0.2000}{2}
= \frac{0.3100}{2}
= \mathbf{0.1550\ \text{Hz}}
$$

$$
\sigma_\text{ctrl}
= \sqrt{\frac{(0.1100 - 0.1550)^2 + (0.2000 - 0.1550)^2}{1}}
= \sqrt{\frac{0.002025 + 0.002025}{1}}
= \sqrt{0.004050}
\approx 0.06364
$$

$$
\text{SEM}_\text{ctrl}
= \frac{0.06364}{\sqrt{2}}
\approx \mathbf{0.04500\ \text{Hz}}
$$

### Condition "drug"  (FOV-C + FOV-D, M = 2)

$$
\bar{x}_\text{drug}
= \frac{0.3100 + 0.3975}{2}
= \frac{0.7075}{2}
= \mathbf{0.35375\ \text{Hz}}
$$

$$
\sigma_\text{drug}
= \sqrt{\frac{(0.3100 - 0.35375)^2 + (0.3975 - 0.35375)^2}{1}}
= \sqrt{\frac{0.001914 + 0.001914}{1}}
= \sqrt{0.003828}
\approx 0.06187
$$

$$
\text{SEM}_\text{drug}
= \frac{0.06187}{\sqrt{2}}
\approx \mathbf{0.04375\ \text{Hz}}
$$

**Final bar-plot values (Metric 1)**

| Condition | Mean (Hz) | SEM (Hz) | FOV dots |
|-----------|-----------|----------|----------|
| ctrl      | 0.1550    | 0.04500  | 0.110, 0.200 |
| drug      | 0.35375   | 0.04375  | 0.310, 0.3975 |

---

## Metric 2 — % Active Cells
*(uses `_aggregate_percentage_data_to_condition_stats` — binomial SEM model)*

```
Condition "ctrl"
  FOV-A: 3 active / 5 total  →  60.00%   n=5
  FOV-B: 4 active / 6 total  →  66.67%   n=6

Condition "drug"
  FOV-C: 7 active / 8 total  →  87.50%   n=8
  FOV-D: 6 active / 7 total  →  85.71%   n=7
```

### Step 1 — ROI → FOV

Just a ratio: $\text{pct}_\text{FOV} = \dfrac{k_\text{active}}{n_\text{total}} \times 100$.

No per-FOV SEM is stored here; the error is handled entirely in Step 2.

### Step 2 — FOV → Condition

$$
\bar{p}_\text{cond}\ (\%)
= \frac{\sum_i n_i\,\text{pct}_i}{\sum_i n_i}
\qquad
\text{SEM}_\text{cond}\ (\%)
= \sqrt{\frac{p\,(1-p)}{N}} \times 100
\quad\text{where } p = \bar{p}_\text{cond}/100,\; N = \sum_i n_i
$$

### Condition "ctrl"

$$
N = 5 + 6 = 11
$$

$$
\bar{p}_\text{ctrl}
= \frac{5 \times 60.00 + 6 \times 66.67}{11}
= \frac{300.00 + 400.02}{11}
= \frac{700.02}{11}
\approx \mathbf{63.64\%}
$$

$$
p = 0.6364, \quad
\text{SEM}_\text{ctrl}
= \sqrt{\frac{0.6364 \times 0.3636}{11}} \times 100
= \sqrt{\frac{0.2313}{11}} \times 100
= \sqrt{0.02103} \times 100
\approx 0.1450 \times 100
\approx \mathbf{14.50\%}
$$

### Condition "drug"

$$
N = 8 + 7 = 15
$$

$$
\bar{p}_\text{drug}
= \frac{8 \times 87.50 + 7 \times 85.71}{15}
= \frac{700.00 + 599.97}{15}
= \frac{1299.97}{15}
\approx \mathbf{86.66\%}
$$

$$
p = 0.8666, \quad
\text{SEM}_\text{drug}
= \sqrt{\frac{0.8666 \times 0.1334}{15}} \times 100
= \sqrt{\frac{0.11563}{15}} \times 100
= \sqrt{0.007709} \times 100
\approx 0.08780 \times 100
\approx \mathbf{8.78\%}
$$

**Final bar-plot values (Metric 2)**

| Condition | Mean (%) | SEM (%) | FOV dots |
|-----------|----------|---------|----------|
| ctrl      | 63.64    | 14.50   | 60.00, 66.67 |
| drug      | 86.66    | 8.78    | 87.50, 85.71 |

---

## Metric 3 — Global Calcium ΔF/F Correlation
*(FOV-level scalar → uses `_aggregate_fov_scalar_to_condition_stats`)*

For pairwise network metrics each FOV yields a **single scalar** (the median
off-diagonal row-mean of the Pearson correlation matrix), and each FOV is
weighted by the number of unique ROI pairs $w = n(n-1)/2$.

```
Condition "ctrl"
  FOV-A: corr = 0.42,  n=4  →  w = 4×3/2 = 6
  FOV-B: corr = 0.55,  n=3  →  w = 3×2/2 = 3

Condition "drug"
  FOV-C: corr = 0.71,  n=3  →  w = 3×2/2 = 3
  FOV-D: corr = 0.68,  n=4  →  w = 4×3/2 = 6
```

### Weighted mean

$$
\bar{x} = \frac{\sum_i w_i\,x_i}{\sum_i w_i}
$$

### Between-FOV weighted SEM

$$
s^2_w = \frac{\sum_i w_i\,(x_i - \bar{x})^2}{W - \sum_i w_i^2 / W}
\qquad W = \sum_i w_i
\qquad
\text{SEM} = \sqrt{s^2_w / M}
\quad (M = \text{number of FOVs})
$$

### Condition "ctrl"  (FOV-A: x=0.42, w=6 · FOV-B: x=0.55, w=3)

$$
W = 6 + 3 = 9
$$

$$
\bar{x}_\text{ctrl}
= \frac{6 \times 0.42 + 3 \times 0.55}{9}
= \frac{2.52 + 1.65}{9}
= \frac{4.17}{9}
\approx \mathbf{0.4633}
$$

$$
\text{denom} = W - \frac{\sum w_i^2}{W} = 9 - \frac{36+9}{9} = 9 - 5 = 4
$$

$$
s^2_w
= \frac{6(0.42-0.4633)^2 + 3(0.55-0.4633)^2}{4}
= \frac{6 \times 0.001878 + 3 \times 0.007524}{4}
= \frac{0.01127 + 0.02257}{4}
= \frac{0.03384}{4}
= 0.008460
$$

$$
\text{SEM}_\text{ctrl} = \sqrt{0.008460 / 2} = \sqrt{0.004230} \approx \mathbf{0.06504}
$$

### Condition "drug"  (FOV-C: x=0.71, w=3 · FOV-D: x=0.68, w=6)

$$
W = 3 + 6 = 9
$$

$$
\bar{x}_\text{drug}
= \frac{3 \times 0.71 + 6 \times 0.68}{9}
= \frac{2.13 + 4.08}{9}
= \frac{6.21}{9}
\approx \mathbf{0.6900}
$$

$$
\text{denom} = 9 - \frac{9+36}{9} = 9 - 5 = 4
$$

$$
s^2_w
= \frac{3(0.71-0.69)^2 + 6(0.68-0.69)^2}{4}
= \frac{3 \times 0.0004 + 6 \times 0.0001}{4}
= \frac{0.0012 + 0.0006}{4}
= \frac{0.0018}{4}
= 0.000450
$$

$$
\text{SEM}_\text{drug} = \sqrt{0.000450 / 2} = \sqrt{0.000225} = \mathbf{0.01500}
$$

**Final bar-plot values (Metric 3)**

| Condition | Mean (r) | SEM (r) | FOV dots |
|-----------|----------|---------|----------|
| ctrl      | 0.4633   | 0.06504 | 0.42, 0.55 |
| drug      | 0.6900   | 0.01500 | 0.71, 0.68 |

---

## Summary of aggregation paths

```
Raw data
  └─ per-ROI scalar (frequency, amplitude, IEI, …)
        │
        ▼  Step 1: μ_FOV = mean(ROIs)
        │
        ▼  Step 2: μ_cond = mean(μ_FOV_1, …, μ_FOV_M)         [unweighted mean]
                    SEM_cond = std(μ_FOVs, ddof=1) / sqrt(M)    [between-FOV SEM]

  └─ per-ROI binary (active / not-active → % Active Cells)
        │
        ▼  Step 1: pct_FOV = k_active / n_total × 100
        │
        ▼  Step 2: p_cond  = Σ(n_i · pct_i) / Σn_i  / 100
                    SEM_cond = sqrt(p · (1-p) / N) × 100        [binomial SEM]

  └─ FOV-level scalar (correlation, synchrony, burst count, …)
        │   (no Step 1; one value per FOV, weighted by n_pairs or 1)
        ▼
        Step 2: x̄_cond   = Σ(w_i · x_i) / Σw_i               [weighted mean]
                 SEM_cond  = sqrt(s²_w / M)                     [weighted between-FOV SEM]
                             s²_w = Σw_i·(x_i-x̄)² / (W - Σw²/W)
```

> **Key design choices**
>
> * Per-ROI metrics treat each FOV as one independent biological replicate —
>   the condition mean is the unweighted mean of FOV means, and the SEM
>   reflects between-FOV variability.
> * Percentage / proportion metrics use the binomial model instead of the
>   standard SEM because the quantity is bounded in [0, 100].
> * Network/pairwise metrics use between-FOV variability (not within-FOV ROI
>   spread) because the scalar is already a population-level summary.
> * Individual FOV means are always overlaid as scatter dots on the bar,
>   making within-condition variability directly visible even when M = 2 FOVs.
