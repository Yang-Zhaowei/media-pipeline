"""Run from a checkout: python -m experiments.clone_prosody --help."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .plan import CONDITIONS, build_matrix
from .runner import repository_state, run_condition, run_experiment, write_json


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Qwen Base clone prosody diagnostic v0 (not a production renderer)")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--voice", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--tts-python", default=sys.executable)
    run.add_argument("--expected-head", required=True)
    run.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    for name in ("analyze", "blind"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "blind":
            command.add_argument("--seed", type=int, default=8108)
    worker = commands.add_parser("_worker", help="internal per-run worker")
    worker.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    synthesis_returned = False
    try:
        if args.command == "plan":
            print(json.dumps(build_matrix(conditions=args.conditions), ensure_ascii=False, indent=2))
            return 0
        if args.command == "_worker":
            spec = json.loads(args.request.read_text(encoding="utf-8"))
            try:
                repository_state(spec["repository"]["head"])
            except Exception as exc:
                write_json(args.request.parent / "report.json", {
                    **spec["run"], "status": "failed", "outputs": [], "listening_wav": None,
                    "failure": {"stage": "worker_preflight", "type": type(exc).__name__,
                                "reason": str(exc), "partial_artifacts": [args.request.name]},
                })
                return 1
            report = run_condition(spec, args.request.parent)
            return 0 if report["status"] == "complete" else 1
        from .analysis import analyze_experiment, create_blind_package
        if args.command == "analyze":
            analysis = analyze_experiment(args.output)
            if analysis["status"] != "complete":
                (args.output / ".complete").unlink(missing_ok=True)
            print(json.dumps({"status": analysis["status"], "report": str(args.output / "analysis.md")}))
            return 0 if analysis["status"] == "complete" else 1
        if args.command == "blind":
            result = create_blind_package(args.output, args.seed)
            print(json.dumps({"status": "complete", "samples": len(result["samples"]),
                              "directory": str(args.output / "listening-blind"),
                              "mapping_path": str(args.output / "listening-mapping.json")}))
            return 0
        manifest = run_experiment(args.output, model=args.model, voice=args.voice,
                                  device=args.device, tts_python=args.tts_python,
                                  expected_head=args.expected_head, conditions=args.conditions)
        synthesis_returned = True
        analysis = analyze_experiment(args.output)
        manifest["synthesis_status"] = manifest["status"]
        manifest["analysis_status"] = analysis["status"]
        if analysis["status"] != "complete":
            manifest["status"] = "incomplete"
        write_json(args.output / "manifest.json", manifest)
        # Successful independent runs can be listened to in an incomplete matrix;
        # both analysis and review sheet explicitly preserve incompleteness.
        if any(r["status"] == "complete" for r in manifest["runs"]):
            create_blind_package(args.output, 8108)
            manifest["blind_package_status"] = "complete"
        else:
            manifest["blind_package_status"] = "unavailable"
        write_json(args.output / "manifest.json", manifest)
        if manifest["status"] == "complete" and analysis["status"] == "complete":
            (args.output / ".complete").write_text("complete\n", encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                          "analysis_status": analysis["status"],
                          "owner_checklist": ["confirm provenance and all selected runs",
                                              "listen blind before opening mapping",
                                              "review identity, continuity, emotional jumps, natural progression",
                                              "record pronunciation, artifacts and notes",
                                              "compare repeat dispersion before interpreting H1-H4"]}, indent=2))
        return 0 if (args.output / ".complete").is_file() else 1
    except Exception as exc:
        if args.command == "analyze":
            (args.output / ".complete").unlink(missing_ok=True)
        if args.command == "run" and synthesis_returned:
            path = args.output / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["status"] = "failed"
            manifest["failure"] = {"stage": "derived_artifacts", "type": type(exc).__name__, "reason": str(exc)}
            (args.output / ".complete").unlink(missing_ok=True)
            write_json(path, manifest)
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
