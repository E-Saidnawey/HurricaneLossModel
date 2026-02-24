# Hurricane Catastrophe Loss Model

A Monte Carlo-based catastrophe model for estimating hurricane-driven insurance losses. The model combines historical HURDAT2 hurricane data with frequency and severity distributions to produce annual loss distributions and standard reinsurance risk metrics.

---

## Model Architecture

```
Historical Data → Frequency Model → Severity Model → Loss Function
                                         ↓
                    Monte Carlo Simulation → Annual Loss Distribution → Reinsurance Layer Analysis
```

---

## Data Source

Historical hurricane data is sourced from NOAA's HURDAT2 database (Atlantic basin, 1851–2023):

> https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html

Raw `.txt` data is parsed into a structured CSV using `Data/hurdat2_txt_to_json.py`. Only post-1950 records are used in model fitting.

---

## Repository Structure

```
├── Data/
│   ├── hurdat2-1851-2023-051124.txt        # Raw HURDAT2 data
│   ├── hurdat2_all_events.csv              # Parsed event-level dataset
│   └── hurdat2_txt_to_json.py              # Parser script
│
├── Scripts/
│   ├── load_data.py                        # Data loading and preprocessing
│   ├── Hurricane_Statistics.py             # Frequency distribution fitting (NegBin vs Poisson)
│   ├── Hurricane_Trend.py                  # Time series decomposition and trend analysis
│   ├── Severity_Model.py                   # Wind speed distribution fitting (lognormal)
│   ├── monte_carlo_simulation.py           # Core simulation engine
│   ├── Simulation_Wrapper.py               # Multi-scenario runner
│   └── Report_Generator.py                 # Output reports and visualizations
│
└── Results/                                # Generated plots and CSVs (not tracked in git)
```

---

## Methodology

### 1. Frequency Model
Annual hurricane counts are fit to a **Negative Binomial distribution** using the Method of Moments. The NegBin is preferred over Poisson when the data is overdispersed (variance > mean), which is typical of hurricane counts.

Parameters are estimated as:

```
r = μ² / (σ² - μ)
p = r / (r + μ)
```

### 2. Severity Model
Peak wind speeds at landfall are extracted from HURDAT2 `record_id == 'L'` entries. A **lognormal distribution** is fit to observed wind speeds (converted from knots to mph) using MLE.

### 3. Loss Function
A power-law damage ratio function maps wind speed to fractional loss:

```
DR(v) = min(α × (v / V₀)^β, 1.0)
```

Default parameters: `V₀ = 74 mph`, `α = 0.0005`, `β = 3`

Event loss is computed as `DR × Exposure`, and annual losses are aggregated across all events in a simulated year.

### 4. Monte Carlo Simulation
For each simulated year (default: **100,000 iterations**):
1. Sample number of hurricanes from the fitted NegBin distribution
2. For each hurricane, sample a wind speed from the lognormal severity model
3. Compute damage ratio and event loss
4. Sum to annual loss (capped at total exposure)

---

## Scenario Analysis

The wrapper runs six frequency scenarios by applying multipliers to the historical mean and variance:

| Scenario | Mean Factor | Variance Factor | Description |
|---|---|---|---|
| Original | 1.0× | 1.0× | Base historical parameters |
| High Mean | 1.2× | 1.0× | +20% frequency |
| Low Mean | 0.8× | 1.0× | −20% frequency |
| High Variance | 1.0× | 2.0× | 2× uncertainty |
| Low Variance | 1.0× | 0.5× | 0.5× uncertainty |
| High Mean + High Variance | 1.2× | 2.0× | Combined stress scenario |

---

## Output Metrics

For each scenario, the model reports:

- **AAL** — Average Annual Loss
- **Percentiles** — 50th, 90th, 95th, 99th, 99.9th
- **Return period losses** — 10, 50, 100, 250, 500 year
- **TVaR** — Tail Value at Risk at 90%, 95%, 99%
- **EP Curve** — Exceedance Probability curve

---

## Setup

```bash
pip install numpy pandas scipy matplotlib statsmodels
```

### Parse raw data
```bash
python Data/hurdat2_txt_to_json.py
```

### Run full simulation
```bash
cd Scripts
python Simulation_Wrapper.py
```

### Run individual analyses
```bash
python Hurricane_Statistics.py     # Frequency distribution fitting
python Hurricane_Trend.py          # Time series analysis
python Severity_Model.py           # Wind speed model
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `scipy` | Statistical distributions and fitting |
| `matplotlib` | Visualization |
| `statsmodels` | Time series decomposition (ADF test, seasonal decompose) |
