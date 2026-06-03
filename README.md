# Hall of Fame Probability Models
 
A collection of logistic regression and random forest models estimating the probability that a baseball player is inducted into the National Baseball Hall of Fame, built for position players, starting pitchers, and relief pitchers. All models are trained on post-integration era players (1945 onward) using data sourced from Baseball Reference.
 
---
 
## Models
 
### Position Players — Random Forest (`HOFForest.ipynb`)
 
The primary position player model. A random forest classifier trained on 677 qualified players (PA ≥ 5,000) with a cross-validated AUC of **0.986**.
 
**Features:**
 
| Feature | Importance | Description |
|---|---|---|
| war | 22.5% | Career WAR — total value accumulated |
| war7 | 21.6% | Peak 7-year WAR — best sustained stretch |
| mvpshare | 17.2% | Career MVP vote share — peak recognition |
| asg | 14.2% | All-Star selections — longevity of excellence |
| winshare | 9.5% | Win Shares — alternative career value metric |
| h | 6.2% | Career hits — offensive durability |
| tob | 5.5% | Times On Base (custom metric) |
| scandal | 2.6% | Character clause flag |
| pos_adj | 0.6% | Positional adjustment |
 
**TOB (Times On Base)** is a custom career metric defined as:
```
TOB = H + (HR × 3) + (2B × 2) + (3B × 2) + BB + SB
```
Rewards all forms of offensive contribution weighted by value without being as context-dependent as RBI.
 
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
 
**Why include scandal as a feature?** Rather than removing PED users and other scandal players from the dataset, the scandal boolean allows the model to explicitly learn that statistically elite players can be excluded from the HOF for non-performance reasons. Barry Bonds and Alex Rodriguez score near 100% on statistical grounds but are correctly classified as non-inducted — the model has learned the character clause is a real factor in induction, not just statistics.
 
**Feature selection methodology:** All available numeric columns were passed to an unconstrained random forest. Features with meaningful importance scores were retained. `jawspercentile` was excluded despite high raw importance because it is a percentile ranking of JAWS — a direct transformation of career value that essentially encodes HOF likelihood. Including it constitutes data leakage.
 
**Why random forest?** A single decision tree is fully interpretable but unstable — small changes in data produce very different trees. The random forest averages 1,000 trees each trained on a random subset of data and features, producing more reliable importance scores and better generalization to new players.
 
---
 
### Position Players — Logistic Regression (`HOFProb.ipynb`)
 
A logistic regression model for position players using a smaller, more interpretable feature set.
 
**Qualification:** 5,000+ career plate appearances
 
**Features:**
- `war7` — Peak 7-year WAR, the foundation of the JAWS system. Captures sustained excellence at a player's best rather than career longevity, which is more representative of true talent
- `tob` — Times On Base, a custom career counting metric defined as `H + (HR × 3) + (2B × 2) + (3B × 2) + BB + SB`. Rewards all forms of offensive contribution weighted by value
- `pos_adj` — Positional adjustment based on FanGraphs defensive spectrum standards. Added explicitly because the model without it significantly undervalued catchers and middle infielders relative to corner outfielders and first basemen
**When to use logistic regression vs random forest:** The logistic regression model produces a clean interpretable formula — each coefficient tells you exactly how much a unit increase in war7 or tob shifts HOF probability. The random forest is more accurate but a black box. Use logistic regression when you want to explain the model's reasoning to a non-technical audience. Use the random forest when you want the most accurate probability estimate.
 
---
 
### Starting Pitchers — Logistic Regression (`HOFProbSP.ipynb`)
 
**Qualification:** 250+ career games started, debut before 2020, debut after 1945
 
**Features:**
- `war7` — Peak 7-year WAR. Chosen over JAWS (career WAR + war7 / 2) specifically because modern pitch count management means starters throw significantly fewer innings than pitchers from earlier eras. Career WAR penalizes contemporary pitchers for organizational decisions outside their control. war7 evaluates peak dominance regardless of era
- `sowe` — A custom metric defined as `SO + W + max(0, (ERA+ - 100) × 15)`. Combines career strikeouts (durability and swing-and-miss ability), career wins (longevity, though imperfect), and a bonus for ERA+ above league average weighted to reward sustained excellence above replacement level. The ERA+ bonus component is clipped at zero so below-average ERA+ seasons do not subtract from the total
**Methodology note:** JAWS was deliberately excluded from this model. As pitchers increasingly operate in a five-inning or six-inning framework, career WAR accumulation diverges from true talent in ways that JAWS partially but not fully corrects. The war7 + SOWE combination evaluates peak value and career contribution through metrics that do not mechanically penalize lower inning totals.
 
---
 
### Relief Pitchers — Logistic Regression (`HOFProbRP.ipynb`)
 
**Qualification:** Debut after 1975, last season before 2020
 
**Features:**
- `sve` — A custom metric defined as `SV + ERA+`. Combines saves (volume of high-leverage opportunities accumulated over a career) with ERA+ (quality of performance, park and era adjusted, 100 is league average). A reliever with 400 saves and a 130 ERA+ scores 530. A reliever with 200 saves and a 160 ERA+ scores 360. The metric rewards both longevity in the closer role and sustained excellence
**Methodology note:** Relief pitcher HOF evaluation is notoriously difficult because the role has changed dramatically since the 1970s and traditional metrics like ERA and wins do not capture closer value well. SVE was designed as a single interpretable number that balances opportunity with quality, avoiding the save-total inflation that unfairly rewards high-volume closers on good teams over dominant shorter-career relievers.
 
---
 
## Key Findings
 
**war and war7 together drive 44% of random forest feature importance** — peak dominance and career value are the primary HOF signals, consistent with how serious analysts evaluate HOF cases.
 
**Analytically justified non-inductees score high.** Kenny Lofton, Dale Murphy, and Keith Hernandez all score above 60% despite not being inducted — widely considered the most statistically legitimate HOF snubs. The model flagging them is a finding, not an error.
 
**Scandal players are the most interesting predictions.** Barry Bonds (career WAR 162.8) and Alex Rodriguez (career WAR 117.4) score near 100% on statistical grounds. The scandal flag correctly downgrades their final probability, showing the model has learned the character clause is a real factor in induction.
 
---
 
## Predicting New Players
 
The random forest notebook includes a prediction cell that runs any player through the trained model. Supply a CSV with the same column structure as `everyhitter.csv` — the model will output HOF probability for every player with 5,000+ career PA.
 
Players who debuted before 1933 are flagged as `Pre-ASG era` in the output. The All-Star game began in 1933, so pre-1933 players have `asg = 0` which the model interprets as a negative signal — systematically underrating players like Babe Ruth and Ty Cobb. This model is designed for the post-1933 era.
 
---
 
## Data
 
All data sourced from [Baseball Reference](https://www.baseball-reference.com). Players marked HOF = 1 are inducted into the Hall of Fame.
 
**Scope:** Post-integration era only. Position players and pitchers who played the majority of their career prior to 1945 are excluded to ensure a consistent talent pool and award availability.
 
---
 
## Requirements
 
```
pandas
numpy
scikit-learn
matplotlib
```
 
---
 
## Repository Structure
 
```
hof-probability-models/
├── README.md
├── data/
│   ├── positionplayers.csv              # Position player training data
│   ├── everyhitter.csv                  # All hitters for prediction
│   ├── startingpitchers.csv             # Starting pitcher data
│   └── reliefpitchers.csv               # Relief pitcher data
├── notebooks/
│   ├── HOFForest.ipynb                  # Random forest — position players
│   ├── HOFProb.ipynb                    # Logistic regression — position players
│   ├── HOFProbSP.ipynb                  # Logistic regression — starting pitchers
│   └── HOFProbRP.ipynb                  # Logistic regression — relief pitchers
└── outputs/
    ├── hof_rf_probabilities.csv             # Random forest HOF probabilities
    ├── everyhitter_hof_probabilities.csv    # Predictions for all hitters
    ├── h_probabilities.csv                  # Logistic regression position player probabilities
    ├── sp_probabilities.csv                 # Starting pitcher probabilities
    ├── rp_probabilities.csv                 # Relief pitcher probabilities
    ├── h_probability_plot.png               # Position player plot
    ├── sp_probability_plot.png              # Starting pitcher plot
    └── rp_probability_plot.png              # Relief pitcher plot
```
 
---
 
## Limitations
 
- **Sample size:** The number of post-integration HOF inductees is relatively small, which limits the statistical power of the models and makes cross-validated metrics more meaningful than single train/test splits
- **Ballot dynamics:** The model predicts HOF worthiness based on career statistics, not BBWAA ballot behavior. Players can be statistically qualified and still not be inducted due to character clause considerations, ballot logjams, or era bias
- **Pre-1933 players:** All-Star data does not exist before 1933. Players who retired before or shortly after 1933 have asg = 0 and are systematically underrated by the random forest model
- **Active players:** Career counting stats are incomplete for active players. HOF probability will increase as players accumulate more seasons
- **Era neutrality:** While ERA+ and war7 partially adjust for era, the model does not fully account for the changing run environment across decades or the structural shift toward bullpen-heavy pitching in the starting pitcher model
- **Relief pitcher role change:** The closer role as currently defined did not exist before the late 1970s. The 1975 debut cutoff partially addresses this but the model is most reliable for players from the modern closer era onward
