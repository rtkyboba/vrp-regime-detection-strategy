"""Regenerate every report's numbers, in dependency order.

    python scripts/run_all.py --quick            # audit + P&L + validation only (a few minutes)
    python scripts/run_all.py                    # everything (regimes and the RNN take a long time)
    python scripts/run_all.py --fetch            # download the pinned raw data first
    python scripts/run_all.py --only audit pnl   # named steps only

Steps run as subprocesses from the repo root and stop at the first failure.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = {
    "fetch": [["scripts/fetch_data.py"]],
    "audit": [["scripts/audit_sharpe.py"]],
    "pnl": [["scripts/pnl_realism.py"]],
    "validate": [["scripts/validate_sharpe.py"]],
    "regimes": [["scripts/regime_comparison.py", "--stage", "all"]],
    # stages are separate so 2016+ is never touched before the architecture is frozen
    "rnn": [["scripts/rnn_signal.py", s] for s in ("dev", "select", "final", "report")],
    "alt": [["scripts/alt_strategies.py"]],
    "combine": [["scripts/combine_signals.py"]],  # needs the regimes and rnn outputs
}
QUICK = ["audit", "pnl", "validate"]
ORDER = ["audit", "pnl", "validate", "regimes", "rnn", "alt", "combine"]


def run(name: str, cmds: list) -> None:
    for cmd in cmds:
        label = " ".join(cmd)
        print(f"\n=== [{name}] {label}", flush=True)
        t0 = time.time()
        rc = subprocess.run([PY, "-W", "ignore", *cmd], cwd=ROOT).returncode
        print(f"=== [{name}] done in {time.time() - t0:.0f}s (exit {rc})", flush=True)
        if rc != 0:
            sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="audit, pnl, validate (validate runs with --quick)")
    ap.add_argument("--fetch", action="store_true", help="download the pinned raw data first")
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run just these steps")
    a = ap.parse_args()

    names = a.only or (QUICK if a.quick else ORDER)
    if a.fetch and "fetch" not in names:
        names = ["fetch", *names]
    for name in names:
        cmds = STEPS[name]
        if a.quick and name == "validate":
            cmds = [[*cmds[0], "--quick"]]
        run(name, cmds)


if __name__ == "__main__":
    main()
