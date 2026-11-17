"""Local experiment commands for the neural mixing console."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="nmc", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(required=True)
    prep = data.add_parser("prepare")
    prep.add_argument("dataset", choices=["medleydb"])
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--inputs", choices=["stems", "raw"], default="stems")
    prep.add_argument("--output", type=Path, default=Path("dmc-data/medleydb/inventory.json"))
    launch = commands.add_parser("run")
    launch.add_argument("recipe", type=Path)
    launch.add_argument("--check", action="store_true")
    launch.add_argument("--runs-dir", type=Path, default=Path("runs"))
    listing = commands.add_parser("runs").add_subparsers(required=True).add_parser("list")
    listing.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    try:
        if args.command == "data":
            from nmc.data.medleydb import prepare

            prepare(args.root, args.inputs, args.output)
        elif args.command == "run":
            from nmc.runner import run

            return run(args.recipe, args.check, args.runs_dir)
        else:
            from nmc.runner import list_runs

            list_runs(args.runs_dir)
    except (ValueError, OSError, KeyError) as err:
        parser.exit(1, f"nmc: {err}\n")
    return 0
