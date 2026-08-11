# Next Observed Indicator (NOI)

**Forecasting when a known threat indicator will next strike a federal agency — shifting cyber defense from reactive cleanup to proactive prevention.**

*Data science project summary • Threat intelligence & proactive cyber defense*

Federal agencies are targeted continuously by malicious network indicators — the IP addresses and domains behind attempted intrusions and attacks. Traditionally, security teams react after a known indicator resurfaces and strikes again. NOI was built to flip that posture: by predicting when a given indicator is likely to attack a federal agency next, defenders can pre-position blocking, monitoring, and analyst attention before the next hit — turning threat intelligence into an early-warning, prevention-first capability.

| **96%** | **1–45** | **Daily** | **4** |
|:---:|:---:|:---:|:---:|
| Model accuracy achieved | Day forecast horizons | Automated refresh | Models in ensemble |

## What It Delivers

NOI produces a daily, decision-ready forecast for every tracked indicator across each federal operating division. For each one it estimates the probability that the indicator will be observed attacking again within 1, 7, 14, 30, and 45 days, and translates that into a plain-language rating — *Highly likely*, *Possibly active*, or *Low confidence*.

Leadership and front-line analysts get a shared, prioritized early-warning picture: which threats are likely to return and warrant defensive action now, and which have gone dormant. In validation, the approach reached up to 96% accuracy in predicting recurrence.

## How It Works

The system consolidates roughly 100 days of daily observation records into a clean history for each indicator, then measures behavioral signals — how recently and how often it has appeared, the typical gap between sightings, and whether its activity is steady or comes in bursts.

Because no single method captures every attack pattern, NOI blends four complementary models into one ensemble score. Combining them makes the forecast markedly more stable and accurate than any model alone, and is what drives the 96% result.

## Business Value

- **Proactive defense** — anticipate attacks before they recur, rather than responding after the fact.
- **Sharper prioritization** — analysts focus on indicators most likely to strike next, not a flat list.
- **Confident retirement** — dormant indicators are aged out on evidence, cutting alert fatigue.
- **Consistency at scale** — the same objective logic runs across every agency, every day.

## Modeling Approach — and Why Each Was Chosen

- **Gradient-Boosted Trees** — chosen to capture the non-linear patterns and interactions between behavioral signals that simpler models miss; the strongest single performer and the primary driver of overall accuracy.
- **Logistic Regression** — chosen as a fast, transparent baseline that keeps the ensemble explainable to non-technical stakeholders and grounds its probability estimates.
- **Weibull Survival Model** — chosen because the core question is fundamentally a "time-to-next-event" problem; it directly models how the gaps between attacks and bursty activity shape when an indicator will return.
- **Exponential / Poisson Rate Model** — chosen as a simple, robust estimate of recurrence based on attack frequency, stabilizing forecasts for indicators with sparse or limited history.

## Technology

| Layer | Stack |
|---|---|
| **Foundation** | Python data-science stack — pandas and NumPy for data engineering and feature preparation |
| **Modeling** | scikit-learn (gradient-boosted trees, logistic regression) and lifelines (Weibull survival analysis), blended with a statistical rate model into a weighted ensemble |
| **Delivery** | Automated daily pipeline producing per-agency forecast tables, with a built-in feedback loop that retrains the models as real attack outcomes are confirmed |

## My Role

I designed, built, and led this project end-to-end — from framing the proactive-defense problem with stakeholders, to engineering the data pipeline, selecting and justifying each modeling approach, and shaping the output so it was directly usable by analysts and leadership.

Beyond the code, my focus was translating a reactive operational gap into a repeatable, prevention-first capability: defining success in the analysts' terms, keeping the models explainable, validating performance to 96% accuracy, and designing the system to keep improving through an outcome-driven feedback loop.

## Repository Contents

- [`observationEventForecasting/EDA.ipynb`](observationEventForecasting/EDA.ipynb) — the exploratory analysis that shaped the system: feed-gap and volume checks, activity concentration, recency-vs-return decay, inter-arrival gap and burstiness analysis, calendar effects, and cross-OpDiv overlap. Each section closes with the design decision it motivated; a summary table maps findings to the choices in the modeling notebook.
- [`observationEventForecasting/NextObservedIndicatorV3.0.ipynb`](observationEventForecasting/NextObservedIndicatorV3.0.ipynb) — the research notebook: data loading, dense panel construction, feature engineering, the four-model ensemble, and the analyst-facing forecast output. Internal file paths are redacted; cell outputs are preserved from the original run.
- [`tests/`](tests) — unit tests validating the model logic and performance. The suite loads the functions directly from the notebook (single source of truth) and verifies feature-engineering correctness, probability validity (bounds, horizon monotonicity), and model performance — each model and the ensemble must rank recurring indicators above dormant ones (AUC ≥ 0.95) on a synthetic cohort with known ground truth. Run with `pip install -r requirements.txt && pytest tests/`.

---

*Next Observed Indicator (NOI) — predictive, proactive threat-indicator forecasting for federal cyber defense. Prepared as a project summary for stakeholder review.*
