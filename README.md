# Hall of Fame Probability — Logistic Regression Models

Three separate logistic regression models estimating the probability that a baseball player is inducted into the National Baseball Hall of Fame, built for position players, starting pitchers, and relief pitchers. All models are trained on post-integration era players (1945 onward) using data sourced from Baseball Reference.

---

## Models

### Position Players

**Qualification:** 5,000+ career plate appearances

**Features:**
- `war7` — Peak 7-year WAR, the foundation of the JAWS system. Captures sustained excellence at a player's best rather than career longevity, which is more representative of true talent
- `tob` — Times On Base, a custom career counting metric defined as `H + (HR × 3) + (2B × 2) + (3B × 2) + BB + SB`. Rewards all forms of offensive contribution weighted by value
- `pos_adj` — Positional adjustment based on FanGraphs defensive spectrum standards. Added explicitly because the model without it significantly undervalued catchers and middle infielders relative to corner outfielders and first basemen

**Positional Adjustments Applied:**

| Position | Adjustment |
|---|---|
| Catcher | +12.5 |
| Shortstop | +7.5 |
| Second Base | +2.5 |
| Center Field | +2.5 |
| Third Base | +2.5 |
| Right Field | 0 |
| Left Field | 0 |
| First Base | 0 |
| DH | 0 |

**Methodology note:** Without positional adjustment the model treated a catcher's war7 and tob identically to a first baseman's. A catcher accumulates those numbers while doing something far more defensively demanding, and HOF voters have historically reflected that. The pos_adj feature explicitly encodes that context.

---

### Starting Pitchers

**Qualification:** 250+ career games started, debut before 2020, debut after 1945

**Features:**
- `war7` — Peak 7-year WAR. Chosen over JAWS (career WAR + war7 / 2) specifically because modern pitch count management means starters throw significantly fewer innings than pitchers from earlier eras. Career WAR penalizes contemporary pitchers for organizational decisions outside their control. war7 evaluates peak dominance regardless of era
- `sowe` — A custom metric defined as `SO + W + max(0, (ERA+ - 100) × 15)`. Combines career strikeouts (durability and swing-and-miss ability), career wins (longevity, though imperfect), and a bonus for ERA+ above league average weighted to reward sustained excellence above replacement level. The ERA+ bonus component is clipped at zero so below-average ERA+ seasons do not subtract from the total

**Methodology note:** JAWS was deliberately excluded from this model. As pitchers increasingly operate in a five-inning or six-inning framework, career WAR accumulation diverges from true talent in ways that JAWS partially but not fully corrects. The war7 + SOWE combination evaluates peak value and career contribution through metrics that do not mechanically penalize lower inning totals.

---

### Relief Pitchers

**Qualification:** Debut after 1975, last season before 2020

**Features:**
- `sve` — A custom metric defined as `SV + ERA+`. Combines saves (volume of high-leverage opportunities accumulated over a career) with ERA+ (quality of performance, park and era adjusted, 100 is league average). A reliever with 400 saves and a 130 ERA+ scores 530. A reliever with 200 saves and a 160 ERA+ scores 360. The metric rewards both longevity in the closer role and sustained excellence

**Methodology note:** Relief pitcher HOF evaluation is notoriously difficult because the role has changed dramatically since the 1970s and traditional metrics like ERA and wins do not capture closer value well. SVE was designed as a single interpretable number that balances opportunity with quality, avoiding the save-total inflation that unfairly rewards high-volume closers on good teams over dominant shorter-career relievers.

---

## Data

All data sourced from [Baseball Reference](https://www.baseball-reference.com). Players marked HOF = 1 are in the HOF includes all inducted position players and pitchers.

**Scope:** Post-integration era only. Position players and pitchers who played the majority of their career prior to 1945 are excluded to ensure a consistent talent pool and award availability.
---

## Requirements

```
pandas
scikit-learn
matplotlib
numpy
```

---

## Repository Structure

```
hof-logistic-regression/
├── README.md
├── data/
│   ├── positionplayers.csv          # Position player data
│   ├── startingpitchers.csv         # Starting pitcher data
│   └── reliefpitchers.csv           # Relief pitcher data
├── notebooks/
│   ├── HOFProb.ipynb           # Position player model
│   ├── HOFProbSP.ipynb         # Starting pitcher model
│   └── HOFProbRP.ipynb         # Relief pitcher model
└── outputs/
    ├── h_probabilities.csv      # Position player HOF probabilities
    ├── sp_probabilities.csv     # Starting pitcher HOF probabilities
    ├── rp_probabilities.csv     # Relief pitcher HOF probabilities
    ├── h_probability_plot.png   # Position player plot
    ├── sp_probability_plot.png  # Starting pitcherr plot
    └── rp_probability_plot.png  # Relief pitcher plot

```

---

## Limitations

- **Sample size:** The number of post-integration HOF inductees is relatively small, which limits the statistical power of the models and makes cross-validated metrics more meaningful than single train/test splits
- **Ballot dynamics:** The model predicts HOF worthiness based on career statistics, not BBWAA ballot behavior. Players can be statistically qualified and still not be inducted due to character clause considerations, ballot logjams, or era bias
- **Era neutrality:** While ERA+ and war7 partially adjust for era, the model does not fully account for the changing run environment across decades or the structural shift toward bullpen-heavy pitching in the starting pitcher model
- **Relief pitcher role change:** The closer role as currently defined did not exist before the late 1970s. The 1975 debut cutoff partially addresses this but the model is most reliable for players from the modern closer era onward
