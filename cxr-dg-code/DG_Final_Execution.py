"""Phase-controlled execution wrapper for the locked DG research suite."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

from cxr_pipeline import (
    DGConfig,
    DG_DOMAINS,
    DG_EXTERNAL_DOMAIN,
    DG_PIPELINE_VERSION,
    PipelineBlocked,
    _dg_coral_selection_id,
    _dg_experiment_id,
    _dg_protocol_hash,
    _dg_required_methods_for_target,
    _dg_slug,
    _dg_smoke_experiment_id,
    aggregate_dg_results,
    evaluate_locked_dg_target,
    initialize_dg_output_tree,
    prepare_dg_manifest,
    quantize_and_evaluate_dg_int8,
    run_dg_gradcam_evaluation,
    run_dg_smoke_test,
    select_dg_coral_lambda,
    sha256_file,
    train_and_lock_dg_experiment,
)


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_cached_json(path: Path, artifact_keys: Sequence[str] = ()) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _json(path)
    for key in artifact_keys:
        artifact = payload.get(key)
        if not artifact or not Path(str(artifact)).is_file():
            return None
    return payload


def _config(args: argparse.Namespace, **gates: Any) -> DGConfig:
    return DGConfig(
        mode=args.mode,
        method=args.method,
        target_domain=args.target,
        seed=args.seed,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        learning_rate=args.learning_rate,
        input_root_override=args.input_root,
        output_root_override=args.output_root,
        audited_source_manifest_override=args.manifest,
        audited_source_manifest_sha256_override=args.manifest_sha256,
        epic_manifest_override=args.epic_manifest,
        epic_root_override=args.epic_root,
        near_duplicate_review_path=args.near_duplicate_review,
        manifest_audit_override=args.manifest_audit,
        run_training=bool(gates.get("run_training", False)),
        allow_target_evaluation=bool(gates.get("allow_target_evaluation", False)),
        model_locked=bool(gates.get("model_locked", False)),
    )


def _manifest(args: argparse.Namespace, config: DGConfig) -> Dict[str, Any]:
    paths = config.resolve_paths()
    expected = Path(paths.manifest_root) / f"DG_MANIFEST_{args.target.replace(' ', '_')}_SEED_{args.seed}.csv"
    audit = Path(paths.report_root) / f"DG_AUDIT_{args.target.replace(' ', '_')}_SEED_{args.seed}.json"
    if expected.is_file() and audit.is_file():
        cached = _json(audit)
        if (
            cached.get("status") == "PASS"
            and cached.get("pipeline_version") == DG_PIPELINE_VERSION
            and cached.get("manifest_sha256") == sha256_file(expected)
        ):
            return cached
    return prepare_dg_manifest(config, check_files=True)


def _experiment_id(args: argparse.Namespace) -> str:
    return _dg_experiment_id(args.method, args.target, args.seed)


def _lock_path(args: argparse.Namespace, experiment_id: str) -> Path:
    root = Path(args.output_root or "/kaggle/working/dg_suite")
    return root / "models" / experiment_id / "MODEL_LOCK.json"


def _lambda_path(args: argparse.Namespace, config: DGConfig) -> Path:
    root = Path(args.output_root or "/kaggle/working/dg_suite")
    return root / "reports" / "coral_lambda_selection" / f"{_dg_coral_selection_id(config)}.json"


def _required_lock_paths(args: argparse.Namespace, current_lock_path: Path) -> List[Path]:
    root = Path(args.output_root or "/kaggle/working/dg_suite")
    paths: List[Path] = [current_lock_path]
    for method in _dg_required_methods_for_target(args.target):
        if method == args.method:
            continue
        else:
            experiment_id = _dg_experiment_id(method, args.target, args.seed)
            paths.append(root / "models" / experiment_id / "MODEL_LOCK.json")
    return list(dict.fromkeys(paths))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.pipeline_version != DG_PIPELINE_VERSION:
        raise PipelineBlocked(
            f"DG pipeline version mismatch: expected {DG_PIPELINE_VERSION}, found {args.pipeline_version}"
        )
    if args.phase == "report":
        root = Path(args.output_root or "/kaggle/working/dg_suite")
        return aggregate_dg_results(root, required_set=args.required_run_set)
    if args.phase == "audit":
        config = _config(args)
        return _manifest(args, config)

    if args.phase == "smoke":
        config = _config(args)
        summary = _manifest(args, config)
        smoke_path = Path(config.resolve_paths().log_root) / _dg_smoke_experiment_id(config) / "smoke_test_result.json"
        cached = _complete_cached_json(smoke_path, ("checkpoint", "saved_model"))
        if cached is not None:
            return cached
        return run_dg_smoke_test(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
        )

    if args.phase == "lambda":
        config = _config(args)
        summary = _manifest(args, config)
        lambda_path = _lambda_path(args, config)
        if lambda_path.is_file():
            return _json(lambda_path)
        return select_dg_coral_lambda(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
        )

    if args.phase == "train":
        config = _config(args, run_training=True)
        selection_path = _lambda_path(args, config) if config.use_coral else None
        if selection_path is not None:
            if not selection_path.is_file():
                raise PipelineBlocked(f"Run the target-specific lambda phase first: {selection_path}")
            selection = _json(selection_path)
            config = replace(config, coral_lambda=float(selection["selected_lambda"]))
        summary = _manifest(args, config)
        experiment_id = _experiment_id(args)
        root = Path(args.output_root or "/kaggle/working/dg_suite")
        smoke_path = root / "logs" / _dg_smoke_experiment_id(config) / "smoke_test_result.json"
        if _complete_cached_json(smoke_path, ("checkpoint", "saved_model")) is None:
            raise PipelineBlocked(f"Run the smoke phase first: {smoke_path}")
        lock_path = Path(root) / "models" / experiment_id / "MODEL_LOCK.json"
        cached = _complete_cached_json(lock_path, ("checkpoint", "saved_model", "training_history_path"))
        expected_protocol_hash = _dg_protocol_hash(config, Path(summary["manifest_path"]))
        if cached is not None and cached.get("protocol_hash") == expected_protocol_hash:
            return cached
        return train_and_lock_dg_experiment(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
            smoke_path,
            experiment_id,
            coral_selection_path=selection_path,
        )

    if args.phase == "evaluate":
        config = _config(args, allow_target_evaluation=True, model_locked=True)
        summary = _manifest(args, config)
        experiment_id = args.experiment_id or _experiment_id(args)
        lock_path = Path(args.lock_path) if args.lock_path else _lock_path(args, experiment_id)
        evaluation_path = Path(config.resolve_paths().log_root) / experiment_id / f"target_evaluation_{_dg_slug(args.target).lower()}.json"
        cached = _complete_cached_json(
            evaluation_path,
            ("prediction_path", "metric_path", "calibration_path", "bootstrap_path"),
        )
        current_lock_sha = sha256_file(lock_path) if lock_path.is_file() else None
        if (
            cached is not None
            and cached.get("manifest_sha256") == summary["manifest_sha256"]
            and cached.get("lock_sha256") == current_lock_sha
        ):
            return cached
        return evaluate_locked_dg_target(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
            lock_path,
            required_lock_paths=_required_lock_paths(args, lock_path),
            expected_methods=_dg_required_methods_for_target(args.target),
        )

    if args.phase == "post":
        config = _config(args, allow_target_evaluation=True, model_locked=True)
        summary = _manifest(args, config)
        experiment_id = args.experiment_id or _experiment_id(args)
        lock_path = Path(args.lock_path) if args.lock_path else _lock_path(args, experiment_id)
        lock = _json(lock_path)
        if config.use_coral:
            config = replace(config, coral_lambda=float(lock.get("coral_lambda", config.coral_lambda)))
        if not args.evaluation_path:
            raise PipelineBlocked("The post phase requires --evaluation-path")
        evaluation_path = Path(args.evaluation_path)
        evaluation = _json(evaluation_path)
        prediction_path = Path(evaluation["prediction_path"])
        gradcam = run_dg_gradcam_evaluation(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
            lock_path,
            prediction_path,
        )
        int8 = quantize_and_evaluate_dg_int8(
            config,
            Path(summary["manifest_path"]),
            summary["manifest_sha256"],
            lock_path,
            evaluation_path,
        )
        return {
            "status": "PASS_POST_ANALYSIS_COMPLETE",
            "experiment_id": experiment_id,
            "target_domain": args.target,
            "gradcam": gradcam,
            "int8": int8,
        }

    raise ValueError(f"Unsupported phase: {args.phase}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("audit", "smoke", "lambda", "train", "evaluate", "post", "report"),
    )
    parser.add_argument("--pipeline-version", default=DG_PIPELINE_VERSION)
    parser.add_argument("--mode", default="AUTO", choices=("AUTO", "LOCAL", "KAGGLE"))
    parser.add_argument("--target", default=DG_EXTERNAL_DOMAIN, choices=(*DG_DOMAINS, DG_EXTERNAL_DOMAIN))
    parser.add_argument("--method", default="A", choices=("A", "B", "C", "D"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--near-duplicate-review")
    parser.add_argument("--manifest")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--epic-manifest")
    parser.add_argument("--epic-root")
    parser.add_argument("--manifest-audit")
    parser.add_argument("--experiment-id")
    parser.add_argument("--lock-path")
    parser.add_argument("--evaluation-path")
    parser.add_argument(
        "--required-run-set",
        default="minimum",
        choices=("available", "minimum", "recommended"),
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "status": "BLOCKED" if isinstance(exc, PipelineBlocked) else "FAILED",
            "error": str(exc),
        }
        print(json.dumps(result, indent=2, default=str))
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
