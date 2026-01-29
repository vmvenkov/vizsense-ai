from dataclasses import dataclass

@dataclass
class ChartFeatures:
    mark: str
    encodings: dict          # raw encoding block
    fields_used: set
    color_categories_field: str | None
    has_aggregate: bool
    y_zero: bool | None      # None if unknown/not applicable

def parse_vegalite_spec(spec: dict) -> ChartFeatures:
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")

    enc = spec.get("encoding", {}) or {}
    fields_used = set()
    color_field = None
    has_aggregate = False

    for ch, v in enc.items():
        if isinstance(v, dict):
            f = v.get("field")
            if f:
                fields_used.add(f)
            if v.get("aggregate"):
                has_aggregate = True
            if ch == "color" and f:
                color_field = f

    # y scale zero check (common for bar)
    y_zero = None
    y = enc.get("y")
    if isinstance(y, dict):
        scale = y.get("scale")
        if isinstance(scale, dict) and "zero" in scale:
            y_zero = bool(scale["zero"])

    return ChartFeatures(
        mark=str(mark) if mark else "unknown",
        encodings=enc,
        fields_used=fields_used,
        color_categories_field=color_field,
        has_aggregate=has_aggregate,
        y_zero=y_zero,
    )
