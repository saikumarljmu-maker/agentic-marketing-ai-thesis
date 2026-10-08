"""
Agentic AI for Autonomous Digital Marketing Campaign Management
Single entry point for the full experimental pipeline.

Run every command from the repository root:

    python main.py prepare                 # 1. Build processed parquet files from raw Criteo data
    python main.py simulate                # 2. Run the 6-policy replay simulation (calls Claude)
    python main.py analyse                 # 3. Statistical analysis of the replay results
    python main.py protect-off             # 4. Reproducibility check: protect-off replay
    python main.py compare-models          # 5. 7-model LLM comparison (Claude + local Ollama models)
    python main.py test                    #    Run the leakage test suite

Use `python main.py <command> --help` for the options of each command.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def _run(args):
    """Run a module or script with the current Python interpreter from the repo root."""
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def cmd_prepare(a):
    sys.path.insert(0, str(SRC))
    from data.criteo_prep import prepare
    kwargs = {}
    if a.raw:
        kwargs["raw_path"] = a.raw
    if a.n_campaigns:
        kwargs["n_campaigns"] = a.n_campaigns
    prepare(**kwargs)
    return 0


def cmd_simulate(a):
    sys.path.insert(0, str(SRC))
    from simulation_loop_v2 import ReplaySimulationV2, LAG_NOTE
    sim = ReplaySimulationV2(
        n_seeds=a.seeds,
        use_llm=not a.no_llm,
        lag_note="" if a.no_lag_note else LAG_NOTE,
        run_name=a.run_name,
    )
    sim.run(budget_fractions=a.budgets)
    return 0


def cmd_analyse(a):
    args = ["-m", "src.evaluation.analyse_v2", "--run", a.run]
    if a.ablation:
        args += ["--ablation", a.ablation]
    return _run(args)


def cmd_protect_off(a):
    return _run(["-m", "src.evaluation.protect_off_replay", "--run", a.run])


def cmd_compare_models(a):
    return _run([str(SRC / "model_comparison.py")])


def cmd_test(a):
    return _run(["-m", "pytest", "tests", "-q"])


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Agentic AI for Autonomous Digital Marketing Campaign Management "
                    "- MSc thesis pipeline (LJMU)",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    s = sub.add_parser("prepare", help="build data/processed/*.parquet from the raw Criteo TSV")
    s.add_argument("--raw", help="path to criteo_attribution_dataset.tsv (default: data/raw/...)")
    s.add_argument("--n-campaigns", type=int, help="optional: sample N whole campaigns")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("simulate", help="run the 6-policy replay simulation")
    s.add_argument("--run-name", default="v2", help="label for output files (default: v2)")
    s.add_argument("--seeds", type=int, default=3, help="random seeds per day (default: 3)")
    s.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.25, 0.125],
                   help="budget fractions of logged spend (default: 0.5 0.25 0.125)")
    s.add_argument("--no-llm", action="store_true",
                   help="skip Claude calls; LLM policy falls back to Thompson sampling")
    s.add_argument("--no-lag-note", action="store_true",
                   help="lag ablation: remove the conversion-lag warning from the prompt")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("analyse", help="bootstrap CIs, calibration, escalation and protection audits")
    s.add_argument("--run", default="v2")
    s.add_argument("--ablation", help="second run name to compare against, e.g. v2_nolag")
    s.set_defaults(func=cmd_analyse)

    s = sub.add_parser("protect-off", help="replay saved LLM decisions with and without protection")
    s.add_argument("--run", default="v2")
    s.set_defaults(func=cmd_protect_off)

    s = sub.add_parser("compare-models", help="run the 7-model scenario comparison")
    s.set_defaults(func=cmd_compare_models)

    s = sub.add_parser("test", help="run the leakage test suite")
    s.set_defaults(func=cmd_test)
    return p


def main():
    args = build_parser().parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
