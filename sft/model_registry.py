"""Model registry helpers for download, training, and evaluation scripts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path("configs/models.json")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    repo_id: str
    local_dir: str
    family: str
    template: str
    parameter_count_b: float
    size_class: str
    requires_auth: bool
    notes: str = ""


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models found in registry: {path}")
    return {
        model_id: ModelSpec(model_id=model_id, **_validate_model(model_id, spec))
        for model_id, spec in models.items()
    }


def get_model(model_id: str, path: Path = DEFAULT_REGISTRY) -> ModelSpec:
    models = load_registry(path)
    try:
        return models[model_id]
    except KeyError as error:
        available = ", ".join(sorted(models))
        raise SystemExit(f"Unknown model id '{model_id}'. Available: {available}") from error


def resolve_model_path(model_id_or_path: str, path: Path = DEFAULT_REGISTRY) -> str:
    models = load_registry(path)
    if model_id_or_path in models:
        return models[model_id_or_path].local_dir
    return model_id_or_path


def resolve_template(model_id_or_template: str, path: Path = DEFAULT_REGISTRY) -> str:
    models = load_registry(path)
    if model_id_or_template in models:
        return models[model_id_or_template].template
    return model_id_or_template


def _validate_model(model_id: str, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid model spec for {model_id}: expected object")
    required = {
        "repo_id",
        "local_dir",
        "family",
        "template",
        "parameter_count_b",
        "size_class",
        "requires_auth",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Invalid model spec for {model_id}: missing {missing}")
    return {
        "repo_id": str(spec["repo_id"]),
        "local_dir": str(spec["local_dir"]),
        "family": str(spec["family"]),
        "template": str(spec["template"]),
        "parameter_count_b": float(spec["parameter_count_b"]),
        "size_class": str(spec["size_class"]),
        "requires_auth": bool(spec["requires_auth"]),
        "notes": str(spec.get("notes", "")),
    }


def _cmd_list(args: argparse.Namespace) -> None:
    models = load_registry(args.registry)
    for model in models.values():
        auth = "auth" if model.requires_auth else "public"
        print(
            f"{model.model_id}\t{model.parameter_count_b:g}B\t"
            f"{model.family}\t{model.template}\t{auth}\t{model.local_dir}"
        )


def _cmd_get(args: argparse.Namespace) -> None:
    model = get_model(args.model_id, args.registry)
    print(json.dumps(model.__dict__, ensure_ascii=False, indent=2))


def _cmd_field(args: argparse.Namespace) -> None:
    model = get_model(args.model_id, args.registry)
    try:
        value = getattr(model, args.field)
    except AttributeError as error:
        fields = ", ".join(model.__dict__)
        raise SystemExit(f"Unknown field '{args.field}'. Available: {fields}") from error
    print(value)


def _cmd_resolve(args: argparse.Namespace) -> None:
    if args.kind == "model":
        print(resolve_model_path(args.value, args.registry))
    elif args.kind == "template":
        print(resolve_template(args.value, args.registry))
    else:
        raise SystemExit("kind must be one of: model, template")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the model registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered models.")
    list_parser.set_defaults(func=_cmd_list)

    get_parser = subparsers.add_parser("get", help="Print one model as JSON.")
    get_parser.add_argument("model_id")
    get_parser.set_defaults(func=_cmd_get)

    field_parser = subparsers.add_parser("field", help="Print one field.")
    field_parser.add_argument("model_id")
    field_parser.add_argument("field")
    field_parser.set_defaults(func=_cmd_field)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve model id values.")
    resolve_parser.add_argument("kind", choices=["model", "template"])
    resolve_parser.add_argument("value")
    resolve_parser.set_defaults(func=_cmd_resolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
