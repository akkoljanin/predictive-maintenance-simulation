# Industrial Automation & Predictive Maintenance — Discrete-Event Simulation

**Master's thesis project · Andrijana Koljanin · Faculty of Technical Sciences, University of Novi Sad · 2026**

A Python/SimPy discrete-event simulation comparing a critical manufacturing station under three **hypothetical** technology scenarios:

1. **A — PLC/SCADA:** conventional monitoring and reactive troubleshooting.
2. **B — IIoT monitoring:** faster detection and diagnosis through sensor monitoring.
3. **C — AI-based predictive maintenance (assumed):** a proportion of failures are predicted in advance and converted to planned maintenance.

> **Important scope note:** Scenario C is a *simulation of assumed predictive-maintenance effectiveness*, not a trained machine-learning failure prediction model. Parameter values reflect illustrative literature-informed assumptions; no plant sensor dataset was used to validate these values.

## Research question

How might technology-enabled reductions in failure detection time and unplanned downtime affect equipment availability and overall equipment effectiveness (OEE)?

## Method and outputs

- A discrete-event simulation using **SimPy**, with **30 independent replications** of a **30-day** simulated period per scenario.
- Metrics: availability, performance, quality, OEE, number of breakdowns, predicted breakdowns and downtime.
- A **one-at-a-time sensitivity analysis** of detection time, repair time, assumed prediction rate, scheduled intervention time, and MTBF.
- CSV tables, JSON results and three automatically generated PNG visualizations.

### Original scenario assumptions

| Parameter | A: PLC/SCADA | B: IIoT | C: AI PdM (assumed) |
|---|---:|---:|---:|
| Mean failure detection time | 45 min | 5 min | 5 min |
| Mean repair time | 60 min | 40 min | 40 min |
| Assumed prediction fraction | 0% | 0% | 60% |
| Scheduled maintenance duration | — | — | 15 min |
| Defect rate | 2.0% | 2.0% | 1.2% |

The common mean time between failures (MTBF) is 480 minutes and mean cycle time is 0.5 minutes.

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/simulation.py
```

Outputs are written to the `results/` folder.

## Project structure

```text
src/simulation.py              # simulation, metrics and sensitivity analysis
results/                       # generated CSV, JSON and charts
requirements.txt               # Python dependencies
README.md                      # explanation and reproduction instructions
.gitignore
```

## Limitations

This is a single-station model rather than a full manufacturing line. All inputs are modeled assumptions (not measured electrical equipment data), so the outputs are scenario comparisons, not forecasted real-world savings. In particular, the failure-event generator uses exponentially distributed intervals while the station can be unavailable, and completed downtime accounting may omit partially elapsed interventions at the simulation boundary. These limitations should be considered before extending the model or drawing operational conclusions.

## Provenance

Reconstructed from the source code in **pp. 50–56** of the master's thesis *A Comparison of Manufacturing Process Automation Technologies* (2026). 