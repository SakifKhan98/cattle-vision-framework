"""CLI entry point for the Phase 9 single-video inference pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Overlay explicit CLI flags on top of the YAML config."""
    if args.video:
        config.setdefault("input", {})["video_path"] = args.video
    if args.output_dir:
        config.setdefault("output", {})["output_root"] = args.output_dir
    if args.conf_thresh is not None:
        config.setdefault("detection", {})["confidence_threshold"] = args.conf_thresh
    if args.cleanup:
        config.setdefault("output", {})["cleanup"] = True
    return config


def _make_printer():
    """Return a progress callback that prints to stdout."""
    _last_was_inline = False

    def callback(event: dict) -> None:
        nonlocal _last_was_inline
        stage = event["stage"]
        total = event["total_stages"]
        name = event["stage_name"]
        frame = event["frame"]
        total_frames = event["total_frames"]
        status = event["status"]

        if status == "running":
            print(
                f"\rStage {stage}/{total}: {name} — frame {frame}/{total_frames}",
                end="",
                flush=True,
            )
            _last_was_inline = True
        elif status == "done":
            if _last_was_inline:
                print()  # end the \r line
                _last_was_inline = False
            print(f"Stage {stage}/{total}: {name} — done ({total_frames} frames)")
        elif status == "error":
            if _last_was_inline:
                print()
                _last_was_inline = False
            print(f"Stage {stage}/{total}: {name} — ERROR", file=sys.stderr)

    return callback


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cattle Vision Framework — single-video inference pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", default=None, help="Root directory for outputs")
    parser.add_argument(
        "--config",
        default="configs/inference/default.yaml",
        help="Path to inference YAML config",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=None,
        help="Detection confidence threshold (overrides config)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete intermediate files after the run",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    config = _load_config(args.config)
    config = _merge_cli_overrides(config, args)

    video_path = config["input"]["video_path"]
    if not Path(video_path).exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    from src.inference.pipeline import run_pipeline

    print(f"Cattle Vision — inference pipeline")
    print(f"  video : {video_path}")
    print(f"  output: {config['output']['output_root']}")
    print(f"  config: {args.config}")
    print()

    run_pipeline(config, progress=_make_printer())

    print("\nDone.")


if __name__ == "__main__":
    main()
