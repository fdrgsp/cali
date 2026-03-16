# Multi-Well Aggregation — Step-by-Step Walkthrough

This document walks through every aggregation that happens in the **Multi Well** tab,
using small concrete numbers so every formula can be verified by hand.

---

## Setup: the example experiment

We have **two conditions** (`ctrl` and `drug`), each with **two wells**.
Each well has **two FOVs** with **2–4 active ROIs per FOV**, making the
FOV → Well averaging step clearly visible.

### Metric 1 — Calcium Peak Frequency (Hz)
*(a per-ROI scalar → uses `_aggregate_fov_data_to_condition_stats`)*

```
Condition "ctrl"
│
├── Well-1
│     ├── FOV-A  (4 ROIs)
│     │     ROI-1 = 0.10 Hz
│     │     ROI-2 = 0.14 Hz
│     │     ROI-3 = 0.12 Hz
│     │     ROI-4 = 0.08 Hz
│     │
│     └── FOV-B  (3 ROIs)
│           ROI-5 = 0.20 Hz
│           ROI-6 = 0.22 Hz
│           ROI-7 = 0.18 Hz
│
└── Well-2
      ├── FOV-C  (2 ROIs)
      │     ROI-8 = 0.30 Hz
      │     ROI-9 = 0.40 Hz
      │
      └── FOV-D  (3 ROIs)
            ROI-10 = 0.50 Hz
            ROI-11 = 0.46 Hz
            ROI-12 = 0.52 Hz

Condition "drug"
│
├── Well-3
│     ├── FOV-E  (3 ROIs)
│     │     ROI-13 = 0.60 Hz
│     │     ROI-14 = 0.70 Hz
│     │     ROI-15 = 0.65 Hz
│     │
│     └── FOV-F  (2 ROIs)
│           ROI-16 = 0.80 Hz
│           ROI-17 = 0.90 Hz
│
└── Well-4
      ├── FOV-G  (4 ROIs)
      │     ROI-18 = 1.00 Hz
      │     ROI-19 = 1.10 Hz
      │     ROI-20 = 0.90 Hz
      │     ROI-21 = 1.00 Hz
      │
      └── FOV-H  (2 ROIs)
            ROI-22 = 1.20 Hz
            ROI-23 = 1.30 Hz
```

---

## Step 1 — ROI → FOV

For each FOV compute the mean of its ROI values:

$$
\mu_\text{FOV} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

### FOV-A  (n = 4, values = [0.10, 0.14, 0.12, 0.08])

$$
\mu_A = \frac{0.10 + 0.14 + 0.12 + 0.08}{4} = \frac{0.44}{4} = \mathbf{0.1100\ \text{Hz}}
$$

### FOV-B  (n = 3, values = [0.20, 0.22, 0.18])

$$
\mu_B = \frac{0.20 + 0.22 + 0.18}{3} = \frac{0.60}{3} = \mathbf{0.2000\ \text{Hz}}
$$

### FOV-C  (n = 2, values = [0.30, 0.40])

$$
\mu_C = \frac{0.30 + 0.40}{2} = \frac{0.70}{2} = \mathbf{0.3500\ \text{Hz}}
$$

### FOV-D  (n = 3, values = [0.50, 0.46, 0.52])

$$
\mu_D = \frac{0.50 + 0.46 + 0.52}{3} = \frac{1.48}{3} \approx \mathbf{0.4933\ \text{Hz}}
$$

### FOV-E  (n = 3, values = [0.60, 0.70, 0.65])

$$
\mu_E = \frac{0.60 + 0.70 + 0.65}{3} = \frac{1.95}{3} = \mathbf{0.6500\ \text{Hz}}
$$

### FOV-F  (n = 2, values = [0.80, 0.90])

$$
\mu_F = \frac{0.80 + 0.90}{2} = \frac{1.70}{2} = \mathbf{0.8500\ \text{Hz}}
$$

### FOV-G  (n = 4, values = [1.00, 1.10, 0.90, 1.00])

$$
\mu_G = \frac{1.00 + 1.10 + 0.90 + 1.00}{4} = \frac{4.00}{4} = \mathbf{1.0000\ \text{Hz}}
$$

### FOV-H  (n = 2, values = [1.20, 1.30])

$$
\mu_H = \frac{1.20 + 1.30}{2} = \frac{2.50}{2} = \mathbf{1.2500\ \text{Hz}}
$$

**Step 1 summary table**

| FOV | Well | Condition | n | μ (Hz)  |
|-----|------|-----------|---|---------|
| A   | 1    | ctrl      | 4 | 0.1100  |
| B   | 1    | ctrl      | 3 | 0.2000  |
| C   | 2    | ctrl      | 2 | 0.3500  |
| D   | 2    | ctrl      | 3 | 0.4933  |
| E   | 3    | drug      | 3 | 0.6500  |
| F   | 3    | drug      | 2 | 0.8500  |
| G   | 4    | drug      | 4 | 1.0000  |
| H   | 4    | drug      | 2 | 1.2500  |

---

## Step 2 — FOV → Well

FOV means within the same well are averaged (unweighted) to produce a single
well mean. Multiple FOVs from the same well are **technical replicates** —
each FOV gets equal weight regardless of how many ROIs it contains.

$$
\mu_\text{Well} = \frac{1}{K}\sum_{j=1}^{K} \mu_{\text{FOV}_j}
$$

### Well-1  (FOV-A: 0.1100, FOV-B: 0.2000)

$$
\mu_{\text{Well-1}} = \frac{0.1100 + 0.2000}{2} = \frac{0.3100}{2} = \mathbf{0.1550\ \text{Hz}}
$$

Note: FOV-A has 4 ROIs and FOV-B has 3 ROIs, but they contribute equally.

### Well-2  (FOV-C: 0.3500, FOV-D: 0.4933)

$$
\mu_{\text{Well-2}} = \frac{0.3500 + 0.4933}{2} = \frac{0.8433}{2} \approx \mathbf{0.4217\ \text{Hz}}
$$

### Well-3  (FOV-E: 0.6500, FOV-F: 0.8500)

$$
\mu_{\text{Well-3}} = \frac{0.6500 + 0.8500}{2} = \frac{1.5000}{2} = \mathbf{0.7500\ \text{Hz}}
$$

### Well-4  (FOV-G: 1.0000, FOV-H: 1.2500)

$$
\mu_{\text{Well-4}} = \frac{1.0000 + 1.2500}{2} = \frac{2.2500}{2} = \mathbf{1.1250\ \text{Hz}}
$$

**Step 2 summary table**

| Well | Condition | FOV means       | Well mean (Hz) |
|------|-----------|-----------------|----------------|
| 1    | ctrl      | 0.1100, 0.2000  | 0.1550         |
| 2    | ctrl      | 0.3500, 0.4933  | 0.4217         |
| 3    | drug      | 0.6500, 0.8500  | 0.7500         |
| 4    | drug      | 1.0000, 1.2500  | 1.1250         |

---

## Step 3 — Well → Condition

Each well is treated as one independent **biological replicate**. The condition
mean is the unweighted mean of well means and the SEM is computed across
well means.

$$
\bar{x}_\text{cond} = \frac{1}{W}\sum_{i=1}^{W} \mu_{\text{Well}_i}
\qquad
\text{SEM}_\text{cond} = \frac{\sigma(\mu_{\text{Well}_1}, \dots, \mu_{\text{Well}_W})}{\sqrt{W}}
\quad\text{where}\quad
\sigma = \text{std}(\text{ddof}=1)
$$

### Condition "ctrl"  (Well-1: 0.1550, Well-2: 0.4217, W = 2)

$$
\bar{x}_\text{ctrl}
= \frac{0.1550 + 0.4217}{2}
= \frac{0.5767}{2}
\approx \mathbf{0.2883\ \text{Hz}}
$$

$$
\sigma_\text{ctrl}
= \sqrt{\frac{(0.1550 - 0.2883)^2 + (0.4217 - 0.2883)^2}{1}}
= \sqrt{\frac{0.01779 + 0.01779}{1}}
= \sqrt{0.03558}
\approx 0.18863
$$

$$
\text{SEM}_\text{ctrl}
= \frac{0.18863}{\sqrt{2}}
\approx \mathbf{0.13333\ \text{Hz}}
$$

### Condition "drug"  (Well-3: 0.7500, Well-4: 1.1250, W = 2)

$$
\bar{x}_\text{drug}
= \frac{0.7500 + 1.1250}{2}
= \frac{1.8750}{2}
= \mathbf{0.9375\ \text{Hz}}
$$

$$
\sigma_\text{drug}
= \sqrt{\frac{(0.7500 - 0.9375)^2 + (1.1250 - 0.9375)^2}{1}}
= \sqrt{\frac{0.03516 + 0.03516}{1}}
= \sqrt{0.07031}
\approx 0.26520
$$

$$
\text{SEM}_\text{drug}
= \frac{0.26520}{\sqrt{2}}
\approx \mathbf{0.18750\ \text{Hz}}
$$

**Final bar-plot values (Metric 1)**

| Condition | Mean (Hz) | SEM (Hz) | Well dots (scatter) |
|-----------|-----------|----------|---------------------|
| ctrl      | 0.2883    | 0.13333  | 0.1550, 0.4217      |
| drug      | 0.9375    | 0.18750  | 0.7500, 1.1250      |

---

## Metric 2 — % Active Cells
*(uses `_aggregate_percentage_data_to_condition_stats` — inter-well SEM)*

```
Condition "ctrl"
│
├── Well-1
│     ├── FOV-A: 3 active / 5 total  →  60.00%
│     └── FOV-B: 5 active / 6 total  →  83.33%
│
└── Well-2
      ├── FOV-C: 2 active / 4 total  →  50.00%
      └── FOV-D: 1 active / 3 total  →  33.33%

Condition "drug"
│
├── Well-3
│     ├── FOV-E: 7 active / 8 total  →  87.50%
│     └── FOV-F: 4 active / 5 total  →  80.00%
│
└── Well-4
      ├── FOV-G: 6 active / 7 total  →  85.71%
      └── FOV-H: 9 active / 10 total →  90.00%
```

### Step 1 — ROI → FOV

Just a ratio: $\text{pct}_\text{FOV} = \dfrac{k_\text{active}}{n_\text{total}} \times 100$.

No per-FOV SEM is stored here; the error is handled entirely in Step 3.

### Step 2 — FOV → Well

FOV percentages within the same well are averaged (unweighted).

| Well | Condition | FOV pcts           | Well mean (%)           |
|------|-----------|--------------------|-------------------------|
| 1    | ctrl      | 60.00, 83.33       | (60.00 + 83.33)/2 = 71.67 |
| 2    | ctrl      | 50.00, 33.33       | (50.00 + 33.33)/2 = 41.67 |
| 3    | drug      | 87.50, 80.00       | (87.50 + 80.00)/2 = 83.75 |
| 4    | drug      | 85.71, 90.00       | (85.71 + 90.00)/2 = 87.86 |

### Step 3 — Well → Condition

$$
\bar{p}_\text{cond}\ (\%)
= \frac{1}{W}\sum_{i=1}^{W} \mu_{\text{Well}_i}
\qquad
\text{SEM}_\text{cond}\ (\%)
= \frac{\text{std}(\text{well means},\ \text{ddof}=1)}{\sqrt{W}}
$$

### Condition "ctrl"  (Well-1: 71.67%, Well-2: 41.67%)

$$
\bar{p}_\text{ctrl}
= \frac{71.67 + 41.67}{2}
= \mathbf{56.67\%}
$$

$$
\text{SEM}_\text{ctrl}
= \frac{\text{std}([71.67, 41.67],\ \text{ddof}=1)}{\sqrt{2}}
= \frac{21.213}{\sqrt{2}}
\approx \mathbf{15.00\%}
$$

### Condition "drug"  (Well-3: 83.75%, Well-4: 87.86%)

$$
\bar{p}_\text{drug}
= \frac{83.75 + 87.86}{2}
= \mathbf{85.80\%}
$$

$$
\text{SEM}_\text{drug}
= \frac{\text{std}([83.75, 87.86],\ \text{ddof}=1)}{\sqrt{2}}
= \frac{2.905}{\sqrt{2}}
\approx \mathbf{2.054\%}
$$

**Final bar-plot values (Metric 2)**

| Condition | Mean (%) | SEM (%) | Well dots        |
|-----------|----------|---------|------------------|
| ctrl      | 56.67    | 15.00   | 71.67, 41.67     |
| drug      | 85.80    | 2.054   | 83.75, 87.86     |

---

## Metric 3 — Global Calcium ΔF/F Correlation
*(FOV-level scalar → uses `_aggregate_fov_scalar_to_condition_stats`)*

For pairwise network metrics each FOV yields a **single scalar** (the median
off-diagonal row-mean of the Pearson correlation matrix), and each FOV is
weighted by the number of unique ROI pairs $w = n(n-1)/2$.

```
Condition "ctrl"
│
├── Well-1
│     ├── FOV-A: corr = 0.42, n=4  →  w = 4×3/2 = 6
│     └── FOV-B: corr = 0.58, n=3  →  w = 3×2/2 = 3
│
└── Well-2
      ├── FOV-C: corr = 0.35, n=5  →  w = 5×4/2 = 10
      └── FOV-D: corr = 0.40, n=2  →  w = 2×1/2 = 1

Condition "drug"
│
├── Well-3
│     ├── FOV-E: corr = 0.71, n=3  →  w = 3×2/2 = 3
│     └── FOV-F: corr = 0.80, n=4  →  w = 4×3/2 = 6
│
└── Well-4
      ├── FOV-G: corr = 0.65, n=6  →  w = 6×5/2 = 15
      └── FOV-H: corr = 0.72, n=3  →  w = 3×2/2 = 3
```

### Step 1 — FOV → Well (weighted mean by n_pairs)

$$
\mu_\text{Well} = \frac{\sum_j w_j\,x_j}{\sum_j w_j}
$$

### Well-1  (FOV-A: x=0.42, w=6 ; FOV-B: x=0.58, w=3)

$$
\mu_{\text{Well-1}}
= \frac{6 \times 0.42 + 3 \times 0.58}{6 + 3}
= \frac{2.52 + 1.74}{9}
= \frac{4.26}{9}
\approx \mathbf{0.4733}
$$

FOV-A (4 ROIs, 6 pairs) contributes more than FOV-B (3 ROIs, 3 pairs)
because it has a more reliable pairwise estimate.

### Well-2  (FOV-C: x=0.35, w=10 ; FOV-D: x=0.40, w=1)

$$
\mu_{\text{Well-2}}
= \frac{10 \times 0.35 + 1 \times 0.40}{10 + 1}
= \frac{3.50 + 0.40}{11}
= \frac{3.90}{11}
\approx \mathbf{0.3545}
$$

### Well-3  (FOV-E: x=0.71, w=3 ; FOV-F: x=0.80, w=6)

$$
\mu_{\text{Well-3}}
= \frac{3 \times 0.71 + 6 \times 0.80}{3 + 6}
= \frac{2.13 + 4.80}{9}
= \frac{6.93}{9}
= \mathbf{0.7700}
$$

### Well-4  (FOV-G: x=0.65, w=15 ; FOV-H: x=0.72, w=3)

$$
\mu_{\text{Well-4}}
= \frac{15 \times 0.65 + 3 \times 0.72}{15 + 3}
= \frac{9.75 + 2.16}{18}
= \frac{11.91}{18}
\approx \mathbf{0.6617}
$$

**Step 1 summary table**

| Well | Condition | FOV scalars      | FOV weights | Well mean |
|------|-----------|------------------|-------------|-----------|
| 1    | ctrl      | 0.42, 0.58       | 6, 3        | 0.4733    |
| 2    | ctrl      | 0.35, 0.40       | 10, 1       | 0.3545    |
| 3    | drug      | 0.71, 0.80       | 3, 6        | 0.7700    |
| 4    | drug      | 0.65, 0.72       | 15, 3       | 0.6617    |

### Step 2 — Well → Condition (unweighted mean ± SEM across wells)

$$
\bar{x}_\text{cond} = \frac{1}{W}\sum_{i=1}^{W} \mu_{\text{Well}_i}
\qquad
\text{SEM}_\text{cond} = \frac{\text{std}(\text{well means},\ \text{ddof}=1)}{\sqrt{W}}
$$

### Condition "ctrl"  (Well-1: 0.4733, Well-2: 0.3545)

$$
\bar{x}_\text{ctrl}
= \frac{0.4733 + 0.3545}{2}
= \frac{0.8279}{2}
\approx \mathbf{0.4139}
$$

$$
\text{SEM}_\text{ctrl}
= \frac{\text{std}([0.4733, 0.3545],\ \text{ddof}=1)}{\sqrt{2}}
= \frac{0.08396}{\sqrt{2}}
\approx \mathbf{0.05938}
$$

### Condition "drug"  (Well-3: 0.7700, Well-4: 0.6617)

$$
\bar{x}_\text{drug}
= \frac{0.7700 + 0.6617}{2}
= \frac{1.4317}{2}
\approx \mathbf{0.7158}
$$

$$
\text{SEM}_\text{drug}
= \frac{\text{std}([0.7700, 0.6617],\ \text{ddof}=1)}{\sqrt{2}}
= \frac{0.07658}{\sqrt{2}}
\approx \mathbf{0.05415}
$$

**Final bar-plot values (Metric 3)**

| Condition | Mean (r) | SEM (r)  | Well dots      |
|-----------|----------|----------|----------------|
| ctrl      | 0.4139   | 0.05938  | 0.4733, 0.3545 |
| drug      | 0.7158   | 0.05415  | 0.7700, 0.6617 |

---

## Summary of aggregation paths

```
Raw data
  └─ per-ROI scalar (frequency, amplitude, IEI, …)
        │
        ▼  Step 1: μ_FOV = mean(ROIs)
        │
        ▼  Step 2: μ_Well = mean(μ_FOV_1, …, μ_FOV_K)          [unweighted mean]
        │
        ▼  Step 3: μ_cond = mean(μ_Well_1, …, μ_Well_W)        [unweighted mean]
                    SEM_cond = std(μ_Wells, ddof=1) / sqrt(W)    [between-well SEM]

  └─ per-ROI binary (active / not-active → % Active Cells)
        │
        ▼  Step 1: pct_FOV = k_active / n_total × 100
        │
        ▼  Step 2: pct_Well = mean(pct_FOV_1, …, pct_FOV_K)    [unweighted mean]
        │
        ▼  Step 3: pct_cond = mean(pct_Well_1, …, pct_Well_W)  [unweighted mean]
                    SEM_cond = std(pct_Wells, ddof=1) / sqrt(W)  [between-well SEM]

  └─ FOV-level scalar (correlation, synchrony, burst count, …)
        │   (no ROI→FOV step; one value per FOV, weighted by n_pairs or 1)
        ▼
        Step 1: μ_Well = Σ(w_j · x_j) / Σw_j                   [weighted mean within well]
        │
        ▼  Step 2: μ_cond = mean(μ_Well_1, …, μ_Well_W)        [unweighted mean]
                    SEM_cond = std(μ_Wells, ddof=1) / sqrt(W)    [between-well SEM]
```

> **Key design choices**
>
> * Per-ROI metrics treat each well as one independent biological replicate —
>   FOVs within the same well are technical replicates averaged into a single
>   well mean. The condition SEM reflects between-well variability.
> * Percentage metrics use between-well SEM (rather than the binomial formula)
>   to avoid pseudo-replication: cells within the same well are not independent,
>   and collecting more ROIs from the same well should not shrink the error bar.
> * Network/pairwise metrics use weighted averaging (by n_pairs) within each
>   well, then unweighted between-well SEM at the condition level.
> * Individual well means are always overlaid as scatter dots on the bar,
>   making within-condition variability directly visible even when W = 2 wells.
