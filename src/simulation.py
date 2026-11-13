"""Reconstructed from the source-code appendix (pp. 50–56) of the 2026 master's thesis.

Formatting was restored from PDF line wrapping. The original scenario assumptions and
simulation logic are preserved. Run: python src/simulation.py
"""
import copy
import json
import math
import random
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy

OUTPUT = Path(__file__).resolve().parents[1] / 'results'
OUTPUT.mkdir(exist_ok=True)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def lognorm_params(mean, cv=0.5):
    """Return mu/sigma of a log-normal distribution given mean and CV."""
    sigma2 = math.log(1 + cv**2)
    return math.log(mean) - sigma2 / 2, math.sqrt(sigma2)


class StationSim:
    """Discrete-event model of one critical production station."""

    def __init__(self, env, p, rng):
        self.env, self.p, self.rng = env, p, rng
        self.parts_ok = 0
        self.parts_bad = 0
        self.unplanned_downtime = 0.0
        self.planned_downtime = 0.0
        self.n_failures = 0
        self.n_predicted = 0
        self.detect_time_total = 0.0
        self.producing = True
        self.proc = env.process(self.working())
        env.process(self.breakdown_source())

    def working(self):
        p = self.p
        while True:
            self.producing = True
            cycle = max(0.05, self.rng.gauss(p['cycle_time'], p['cycle_time'] * 0.1))
            try:
                yield self.env.timeout(cycle)
                if self.rng.random() < p['defect_rate']:
                    self.parts_bad += 1
                else:
                    self.parts_ok += 1
            except simpy.Interrupt:
                self.producing = False
                yield from self.handle_failure()
                continue

    def breakdown_source(self):
        while True:
            ttf = self.rng.expovariate(1.0 / self.p['mtbf'])
            yield self.env.timeout(ttf)
            if self.proc.is_alive and self.producing:
                self.proc.interrupt()

    def handle_failure(self):
        p = self.p
        self.n_failures += 1
        predicted = self.rng.random() < p.get('predict_frac', 0.0)
        if predicted:
            self.n_predicted += 1
            dur = p['sched_maint_time']
            yield self.env.timeout(dur)
            self.planned_downtime += dur
        else:
            mu_d, sig_d = lognorm_params(p['detect_mean'], p['detect_cv'])
            mu_r, sig_r = lognorm_params(p['repair_mean'], p['repair_cv'])
            detect = self.rng.lognormvariate(mu_d, sig_d)
            repair = self.rng.lognormvariate(mu_r, sig_r)
            dur = detect + repair
            yield self.env.timeout(dur)
            self.unplanned_downtime += dur
            self.detect_time_total += detect


def run_single(params, sim_time, seed):
    """Run one replication and calculate equipment KPIs."""
    rng = random.Random(seed)
    env = simpy.Environment()
    station = StationSim(env, params, rng)
    env.run(until=sim_time)
    tt = sim_time
    pd_ = station.planned_downtime
    ppt = tt - pd_
    ud = station.unplanned_downtime
    ot = ppt - ud
    total_parts = station.parts_ok + station.parts_bad
    availability = ot / ppt if ppt > 0 else 0
    performance = (params['cycle_time'] * total_parts) / ot if ot > 0 else 0
    performance = min(performance, 1.0)
    quality = station.parts_ok / total_parts if total_parts > 0 else 0
    oee = availability * performance * quality
    return dict(
        availability=availability, performance=performance, quality=quality, oee=oee,
        unplanned_downtime_pct=ud / tt * 100, planned_downtime_pct=pd_ / tt * 100,
        n_failures=station.n_failures, n_predicted=station.n_predicted,
        good_parts=station.parts_ok, bad_parts=station.parts_bad, total_parts=total_parts,
        mttr_observed=(ud / (station.n_failures - station.n_predicted))
        if station.n_failures - station.n_predicted > 0 else 0,
    )


def run_replications(params, sim_time=43200, n_reps=30, base_seed=1000):
    """Aggregate independent runs and approximate 95% confidence intervals."""
    results = [run_single(params, sim_time, base_seed + i) for i in range(n_reps)]
    agg = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        mean = st.mean(vals)
        sd = st.pstdev(vals)
        ci95 = 1.96 * sd / math.sqrt(len(vals))
        agg[key] = (mean, ci95)
    return agg, results


COMMON = dict(cycle_time=0.5, mtbf=480, defect_rate=0.02)
SCENARIOS = {
    'A_PLC_SCADA': dict(COMMON, detect_mean=45, detect_cv=0.6,
                        repair_mean=60, repair_cv=0.5, predict_frac=0.0, sched_maint_time=0),
    'B_IIoT': dict(COMMON, detect_mean=5, detect_cv=0.5,
                   repair_mean=40, repair_cv=0.5, predict_frac=0.0, sched_maint_time=0),
    'C_AI_PdM': dict(COMMON, detect_mean=5, detect_cv=0.5,
                     repair_mean=40, repair_cv=0.5, predict_frac=0.6,
                     sched_maint_time=15, defect_rate=0.012),
}
SCENARIO_LABELS = {'A_PLC_SCADA': 'A - PLC/SCADA',
                   'B_IIoT': 'B - IIoT monitoring', 'C_AI_PdM': 'C - AI predictive maintenance'}
SIM_TIME = 43200
N_REPS = 30


def main():
    base_results = {}
    for name, params in SCENARIOS.items():
        agg, _ = run_replications(params, SIM_TIME, N_REPS)
        base_results[name] = agg
    rows = []
    for name, agg in base_results.items():
        rows.append({
            'Scenario': SCENARIO_LABELS[name],
            'Availability (%)': round(agg['availability'][0] * 100, 2),
            'Performance (%)': round(agg['performance'][0] * 100, 2),
            'Quality (%)': round(agg['quality'][0] * 100, 2),
            'OEE (%)': round(agg['oee'][0] * 100, 2),
            'Unplanned downtime (%)': round(agg['unplanned_downtime_pct'][0], 2),
            'Failures': round(agg['n_failures'][0], 1),
            'Predicted failures': round(agg['n_predicted'][0], 1),
        })
    df_base = pd.DataFrame(rows)
    df_base.to_csv(OUTPUT / 'baseline_results.csv', index=False)
    print('Baseline results:\n', df_base.to_string(index=False))

    scenarios_order = list(SCENARIOS)
    colors = ['#c0504d', '#4f81bd', '#9bbb59']
    oee_vals = [base_results[s]['oee'][0] * 100 for s in scenarios_order]
    labels = [SCENARIO_LABELS[s] for s in scenarios_order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, oee_vals, color=colors, edgecolor='black', linewidth=0.6)
    ax.set_ylabel('OEE (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Overall equipment effectiveness (OEE) by scenario')
    for bar, value in zip(bars, oee_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                f'{value:.2f}%', ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'oee_by_scenario.png', dpi=200)
    plt.close(fig)

    metrics = ['Availability', 'Performance', 'Quality']
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, scenario in enumerate(scenarios_order):
        agg = base_results[scenario]
        vals = [agg['availability'][0] * 100, agg['performance'][0] * 100,
                agg['quality'][0] * 100]
        ax.bar(x + (i - 1) * width, vals, width,
               label=SCENARIO_LABELS[scenario], color=colors[i],
               edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('%')
    ax.set_ylim(0, 105)
    ax.set_title('OEE decomposition')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'oee_components.png', dpi=200)
    plt.close(fig)

    sensitivity_specs = {
        'A_PLC_SCADA': [('detect_mean', 30, 45, 60), ('repair_mean', 45, 60, 90)],
        'B_IIoT': [('detect_mean', 2, 5, 10), ('repair_mean', 25, 40, 55)],
        'C_AI_PdM': [('predict_frac', 0.4, 0.6, 0.8),
                     ('sched_maint_time', 10, 15, 25), ('repair_mean', 25, 40, 55)],
    }
    sens_rows = []
    sens_results = {}
    for scenario, specs in sensitivity_specs.items():
        sens_results[scenario] = {}
        for param_name, low, base, high in specs:
            row = {}
            for label, val in [('low', low), ('baseline', base), ('high', high)]:
                p = copy.deepcopy(SCENARIOS[scenario])
                p[param_name] = val
                agg, _ = run_replications(p, SIM_TIME, N_REPS)
                row[label] = agg['oee'][0] * 100
            sens_results[scenario][param_name] = row
            sens_rows.append({'Scenario': SCENARIO_LABELS[scenario],
                              'Parameter': param_name,
                              'OEE low (%)': round(row['low'], 2),
                              'OEE baseline (%)': round(row['baseline'], 2),
                              'OEE high (%)': round(row['high'], 2)})
    pd.DataFrame(sens_rows).to_csv(OUTPUT / 'parameter_sensitivity.csv', index=False)

    mtbf_values = [360, 480, 600]
    mtbf_rows = []
    mtbf_chart_data = {s: [] for s in scenarios_order}
    for mtbf in mtbf_values:
        for scenario in scenarios_order:
            p = copy.deepcopy(SCENARIOS[scenario])
            p['mtbf'] = mtbf
            agg, _ = run_replications(p, SIM_TIME, N_REPS)
            mtbf_chart_data[scenario].append(agg['oee'][0] * 100)
        mtbf_rows.append({'MTBF (min)': mtbf, **{
            SCENARIO_LABELS[s] + ' OEE (%)': round(mtbf_chart_data[s][-1], 2)
            for s in scenarios_order}})
    pd.DataFrame(mtbf_rows).to_csv(OUTPUT / 'mtbf_sensitivity.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^']
    for i, scenario in enumerate(scenarios_order):
        ax.plot(mtbf_values, mtbf_chart_data[scenario], marker=markers[i],
                linewidth=2, label=SCENARIO_LABELS[scenario], color=colors[i])
    ax.set_xlabel('Mean time between failures (MTBF, min)')
    ax.set_ylabel('OEE (%)')
    ax.set_title('Sensitivity of scenario ranking to MTBF')
    ax.set_ylim(60, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / 'mtbf_sensitivity.png', dpi=200)
    plt.close(fig)

    export = {
        'baseline_results': {
            SCENARIO_LABELS[key]: {metric: value[0] for metric, value in agg.items()}
            for key, agg in base_results.items()
        },
        'parameter_sensitivity': sens_results,
        'mtbf_sensitivity': {str(m): row for m, row in zip(mtbf_values, mtbf_rows)},
    }
    with open(OUTPUT / 'simulation_results.json', 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f'\nResults saved to {OUTPUT}')


if __name__ == '__main__':
    main()
