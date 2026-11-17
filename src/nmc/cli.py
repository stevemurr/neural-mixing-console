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
    prep.add_argument(
        "--inputs", choices=["raw", "stems"], default="raw", help="input tracks (default: raw)"
    )
    prep.add_argument("--output", type=Path, help="default: dmc-data/medleydb/<inputs>.json")
    mix = commands.add_parser("mix", help="predict song-wide EQ and render a raw tracks folder")
    mix.add_argument("tracks_dir", type=Path)
    mix.add_argument("--checkpoint", type=Path)
    mix.add_argument(
        "--out-dir", type=Path, help="new output folder (default: sibling <tracks>-mix)"
    )
    render = commands.add_parser("render", help="replay editable mix settings")
    render.add_argument("tracks_dir", type=Path)
    render.add_argument("settings", type=Path)
    render.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser("run")
    launch.add_argument("recipe", type=Path)
    launch.add_argument("--check", action="store_true")
    launch.add_argument("--runs-dir", type=Path, default=Path("runs"))
    listen = commands.add_parser("audition", help="listen to experiment runs in your browser")
    listen.add_argument("--runs-dir", type=Path, default=Path("runs"))
    listen.add_argument(
        "--host", default="127.0.0.1", help="IPv4 bind address (0.0.0.0 for network access)"
    )
    listen.add_argument("--port", type=int, default=8766)
    listing = commands.add_parser("runs").add_subparsers(required=True).add_parser("list")
    listing.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    try:
        if args.command == "audition":
            from nmc.audition import serve

            serve(args.runs_dir, args.port, host=args.host)
        elif args.command == "mix":
            from nmc.mix import DEFAULT_CHECKPOINT, mix_folder

            root = args.tracks_dir.resolve()
            mix_folder(
                root,
                args.checkpoint or DEFAULT_CHECKPOINT,
                args.out_dir or root.with_name(root.name + "-mix"),
            )
        elif args.command == "render":
            from nmc.mix import replay

            replay(args.tracks_dir, args.settings, args.output)
        elif args.command == "data":
            from nmc.data.medleydb import prepare

            output = args.output or Path("dmc-data/medleydb") / f"{args.inputs}.json"
            prepare(args.root, args.inputs, output)
        elif args.command == "run":
            from nmc.runner import run

            return run(args.recipe, args.check, args.runs_dir)
        else:
            from nmc.runner import list_runs

            list_runs(args.runs_dir)
    except (ValueError, OSError, KeyError) as err:
        parser.exit(1, f"nmc: {err}\n")
    return 0
