from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class GSONUtility:
    @staticmethod
    def print_instance_as_json(instance: T) -> str:
        return json.dumps(instance, indent=2, allow_nan=True, default=_json_default)

    @staticmethod
    def save_instance_to_json(instance: T, file_name: str) -> None:
        with Path(file_name).open("w", encoding="utf-8") as handle:
            json.dump(instance, handle, indent=2, allow_nan=True, default=_json_default)

    @staticmethod
    def retrieve_json_instance(file_name: str):
        try:
            with Path(file_name).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except OSError:
            return None

    @staticmethod
    def retrieve_json_instance_typed(file_name: str, type_: type[T]) -> T | None:
        data = GSONUtility.retrieve_json_instance(file_name)
        if data is None:
            return None
        try:
            if hasattr(type_, "__annotations__"):
                return type_(**data)  # type: ignore[misc]
            return type_(data)  # type: ignore[call-arg]
        except Exception:
            return None


def _json_default(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)
