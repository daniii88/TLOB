#!/usr/bin/env python3
"""Thin training/backtest runner for lob_model (TLOB-based).

This keeps a profile-driven CLI workflow similar to the existing
tft-live-strategy-selector_v2 tooling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "profiles" / "btc_tlob_fast.json"


def load_profile(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def profile_overrides_to_hydra_args(overrides: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in overrides.items():
        args.append(f"{key}={normalize_value(value)}")
    return args


def parse_cli_overrides(raw_overrides: List[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(
                f"invalid --set value `{raw}`; expected KEY=VALUE (Hydra style)"
            )
        out.append(raw)
    return out


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def cmd_train(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    train_cfg = profile.get("train", {})
    model = train_cfg.get("model", "tlob")
    dataset = train_cfg.get("dataset", "btc")
    overrides = train_cfg.get("overrides", {})
    hydra_args = profile_overrides_to_hydra_args(overrides)
    hydra_args.extend(parse_cli_overrides(args.set))

    cmd = [
        args.python,
        "main.py",
        f"+model={model}",
        f"+dataset={dataset}",
        "hydra.job.chdir=False",
    ]
    cmd.extend(hydra_args)
    return run_cmd(cmd, ROOT)


def cmd_backtest(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    backtest_cfg = profile.get("backtest", {})
    script = backtest_cfg.get("script", "run_backtesting.py")
    script_args = [str(v) for v in backtest_cfg.get("args", [])]
    cmd = [args.python, script]
    cmd.extend(script_args)
    return run_cmd(cmd, ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile-based runner for lob_model (TLOB training/backtest)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="run model training via TLOB main.py")
    p_train.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE),
        help="JSON profile path",
    )
    p_train.add_argument(
        "--python",
        default=sys.executable,
        help="python executable to use",
    )
    p_train.add_argument(
        "--set",
        action="append",
        default=[],
        help="extra Hydra override in KEY=VALUE format; repeatable",
    )
    p_train.set_defaults(func=cmd_train)

    p_backtest = sub.add_parser(
        "backtest", help="run backtesting script defined in profile"
    )
    p_backtest.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE),
        help="JSON profile path",
    )
    p_backtest.add_argument(
        "--python",
        default=sys.executable,
        help="python executable to use",
    )
    p_backtest.set_defaults(func=cmd_backtest)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
