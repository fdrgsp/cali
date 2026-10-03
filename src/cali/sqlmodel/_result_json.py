"""Acyclic JSON snapshots of normalized products and their provenance."""

import json
import math
from functools import lru_cache
from types import GenericAlias
from typing import Any, Self, cast

from pydantic import ConfigDict, SerializationInfo, TypeAdapter, model_serializer
from sqlmodel import SQLModel


def _relations(name: str) -> dict[str, tuple[type[SQLModel], bool]]:
    # Resolve after all ORM classes are registered; never serialize back-pointers.
    from ._model import DataAnalysis, FOVAnalysis, Traces
    from ._spike_analysis import SpikeAnalysis
    from ._spike_fov_analysis import SpikeFOVAnalysis
    from ._trace_provenance import ExtractionFrameWindow, SpikeInferenceRun, SpikeTrace

    registry: dict[str, dict[str, tuple[type[SQLModel], bool]]] = {
        Traces.__name__: {
            "extraction_frame_window": (ExtractionFrameWindow, False),
            "spike_traces": (SpikeTrace, True),
        },
        SpikeTrace.__name__: {"inference_run": (SpikeInferenceRun, False)},
        DataAnalysis.__name__: {"spike_analyses": (SpikeAnalysis, True)},
        SpikeAnalysis.__name__: {"spike_trace": (SpikeTrace, False)},
        FOVAnalysis.__name__: {"spike_analyses": (SpikeFOVAnalysis, True)},
        SpikeFOVAnalysis.__name__: {"inference_run": (SpikeInferenceRun, False)},
    }
    return registry.get(name, {})


def _prepare(
    name: str,
    data: dict[str, Any],
    *,
    clone: bool = False,
    strict: bool | None = None,
    context: Any = None,
) -> dict[str, Any]:
    result = dict(data)
    for relation, (target, many) in _relations(name).items():
        if relation not in result:
            continue
        value = result[relation]
        if many and not isinstance(value, (list, tuple)):
            raise ValueError(f"{relation} must be a list of normalized records.")
        items = value if many else [value]
        parsed: list[SQLModel | None] = []
        for item in items:
            if item is None and not many:
                parsed.append(None)
            elif isinstance(item, target) and not clone:
                parsed.append(item)
            elif isinstance(item, (dict, target)):
                if isinstance(item, SQLModel):
                    item = {
                        field: getattr(item, field)
                        for field in (
                            *type(item).model_fields,
                            *_relations(type(item).__name__),
                        )
                    }
                parsed.append(
                    target.model_validate(item, strict=strict, context=context)
                )
            else:
                raise ValueError(f"{relation} must contain {target.__name__} records.")
        result[relation] = parsed if many else parsed[0]
    return result


def _validate_snapshot(obj: SQLModel) -> None:
    from ._model import DataAnalysis, FOVAnalysis, Traces
    from ._spike_analysis import SpikeAnalysis
    from ._spike_fov_analysis import SpikeFOVAnalysis
    from ._trace_provenance import SpikeInferenceRun, SpikeTrace

    name = type(obj).__name__
    run: SpikeInferenceRun | None
    if isinstance(obj, SpikeTrace):
        run = obj.inference_run
        if run is None:
            raise ValueError(
                "A spike trace JSON snapshot requires inference provenance."
            )
        start, stop = obj.valid_start, obj.resolved_valid_stop
        if not 0 <= start <= stop <= len(obj.values):
            raise ValueError("Spike valid interval must lie within the stored array.")
        method, units = run.method, run.units
    elif isinstance(obj, (SpikeAnalysis, SpikeFOVAnalysis)):
        method, units = obj.method, obj.units
        if isinstance(obj, SpikeAnalysis):
            run = obj.spike_trace.inference_run if obj.spike_trace is not None else None
        else:
            run = obj.inference_run
        if run is not None and (run.method != method or run.units != units):
            raise ValueError("Spike result and inference provenance must match.")
    else:
        method, units = None, None
    if method is not None and (
        method not in {"oasis", "cascade"}
        or units != {"oasis": "a.u.", "cascade": "spikes/frame"}[method]
    ):
        raise ValueError("Spike method and units must match.")
    if isinstance(obj, SpikeAnalysis) and obj.provenance_source != "legacy_unresolved":
        modes = {"oasis": {"global", "multiplier"}, "cascade": {"global", "cascade_ap"}}
        mode = obj.threshold_mode
        if mode is not None and mode not in modes[obj.method]:
            raise ValueError("Threshold mode is incompatible with the spike method.")
        oasis = ("suprathreshold_sample_rate_hz", "suprathreshold_rising_edge_rate_hz")
        cascade = (
            "expected_spike_rate_hz",
            "expected_spike_count",
            "suprathreshold_excursion_rate_hz",
        )
        if any(
            getattr(obj, field) is not None
            for field in (oasis if method == "cascade" else cascade)
        ):
            raise ValueError("Spike metrics are incompatible with the spike method.")
        for field in ("threshold", *oasis, *cascade):
            value = getattr(obj, field)
            # Preserve the applied OASIS sparse-input disable-detection sentinel.
            if (
                field == "threshold"
                and obj.method == "oasis"
                and obj.threshold_mode == "multiplier"
                and value == math.inf
            ):
                continue
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Spike metrics must be finite and non-negative.")
    if isinstance(obj, (Traces, DataAnalysis, FOVAnalysis)):
        children = getattr(
            obj, "spike_traces" if name == "Traces" else "spike_analyses"
        )
        methods = [
            child.inference_run.method if name == "Traces" else child.method
            for child in children
        ]
        if len(methods) != len(set(methods)):
            raise ValueError("Duplicate spike methods in JSON snapshot.")
        parent_fk = {
            "Traces": "trace_id",
            "DataAnalysis": "data_analysis_id",
            "FOVAnalysis": "fov_analysis_id",
        }[name]
        for child in children:
            if (
                obj.id is not None
                and getattr(child, parent_fk) is not None
                and getattr(child, parent_fk) != obj.id
            ):
                raise ValueError("JSON child and parent IDs must match.")
            for field in ("analysis_result_id", "fov_id"):
                owner_id = getattr(obj, field, None)
                child_id = getattr(child, field, None)
                if (
                    owner_id is not None
                    and child_id is not None
                    and owner_id != child_id
                ):
                    raise ValueError("JSON child and parent ownership must match.")
    for relation, (_, many) in _relations(name).items():
        if many:
            continue
        target = getattr(obj, relation)
        fk = {"inference_run": "spike_inference_run_id"}.get(relation, f"{relation}_id")
        if (
            target is not None
            and target.id is not None
            and getattr(obj, fk, None) is not None
            and getattr(obj, fk) != target.id
        ):
            raise ValueError("JSON relationship and foreign-key IDs must match.")


class ResultJSON(SQLModel):
    """Preserve normalized children without traversing ROI/run back-pointers."""

    # Existing CCG matrices can contain infinite diagonal z-scores. Preserve the
    # same JSON constants as the database encoder instead of replacing them by null.
    model_config = {"ser_json_inf_nan": "constants"}

    def __init__(self, **data: Any) -> None:
        super().__init__(**_prepare(type(self).__name__, data))

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: Any = None,
        update: dict[str, Any] | None = None,
    ) -> Self:
        relations = _relations(cls.__name__)
        if isinstance(obj, cls) or (from_attributes and not isinstance(obj, dict)):
            obj = {
                name: getattr(obj, name)
                for name in (*cls.model_fields, *relations)
                if hasattr(obj, name)
            }
        if not isinstance(obj, dict):
            validated: Self = super().model_validate(
                obj,
                strict=strict,
                from_attributes=from_attributes,
                context=context,
                update=update,
            )
            return validated
        data = _prepare(
            cls.__name__,
            {**obj, **(update or {})},
            clone=True,
            strict=strict,
            context=context,
        )
        legacy_ordering = (
            cls.__name__ == "FOVAnalysis"
            and "active_roi_labels" in data
            and "calcium_active_roi_labels" not in data
        )
        scalar_data = {
            key: value for key, value in data.items() if key not in relations
        }
        if legacy_ordering:
            scalar_data["calcium_active_roi_labels"] = data["active_roi_labels"]
        result: Self = super().model_validate(
            scalar_data,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
        )
        # Run legacy constructors on already validated scalar fields. This keeps
        # their explicit synthetic provenance while JSON relationships are cloned.
        prototype_data = {
            **data,
            **{field: getattr(result, field) for field in cls.model_fields},
        }
        if legacy_ordering:
            prototype_data["active_roi_labels"] = prototype_data.pop(
                "calcium_active_roi_labels"
            )
        prototype = cls(**prototype_data)
        parsed = _prepare(
            cls.__name__,
            {relation: getattr(prototype, relation) for relation in relations},
            clone=True,
            strict=strict,
            context=context,
        )
        for relation, value in parsed.items():
            setattr(result, relation, value)
        _validate_snapshot(result)
        return result

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        context: Any = None,
    ) -> Self:
        return cls.model_validate(json.loads(json_data), strict=strict, context=context)

    @model_serializer(mode="wrap")
    def serialize_result(self, handler: Any, info: SerializationInfo) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for relation, (target, many) in _relations(type(self).__name__).items():
            if info.include is not None and relation not in info.include:
                continue
            if info.exclude is not None and relation in info.exclude:
                exclusion: Any = (
                    info.exclude[relation] if isinstance(info.exclude, dict) else True
                )
                if exclusion is True or exclusion is Ellipsis:
                    continue
            value = getattr(self, relation)
            if value is None:
                if not info.exclude_none:
                    data[relation] = None
                continue
            include: Any = (
                info.include.get(relation) if isinstance(info.include, dict) else None
            )
            exclude: Any = (
                info.exclude.get(relation) if isinstance(info.exclude, dict) else None
            )
            data[relation] = _adapter(cast("Any", target), many).dump_python(
                value,
                mode="json" if info.mode_is_json() else "python",
                include=include,
                exclude=exclude,
                by_alias=info.by_alias,
                exclude_none=info.exclude_none,
                exclude_defaults=info.exclude_defaults,
                exclude_unset=info.exclude_unset,
                serialize_as_any=info.serialize_as_any,
                context=info.context,
            )
        return data


@lru_cache
def _adapter(target: type[SQLModel], many: bool) -> TypeAdapter[Any]:
    if many:
        return TypeAdapter(
            GenericAlias(list, target), config=ConfigDict(ser_json_inf_nan="constants")
        )
    return TypeAdapter(target)
