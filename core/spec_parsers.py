from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def parse_vegalite(spec: Dict[str, Any], data_profile: Any) -> Dict[str, Any]:
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    mark = str(mark).lower() if mark else "unknown"

    enc = spec.get("encoding") or {}
    title = spec.get("title")
    if isinstance(title, dict):
        title = title.get("text")

    # detect value axis and zero baseline
    value_axis_zero = None
    value_axis = None
    for ch in ("y", "x"):
        e = enc.get(ch)
        if isinstance(e, dict) and str(e.get("type", "")).lower() in ("quantitative", "q"):
            value_axis = ch
            sc = e.get("scale")
            if isinstance(sc, dict) and "zero" in sc:
                value_axis_zero = bool(sc["zero"])
            break

    # color cardinality
    color_card = 0
    color = enc.get("color")
    if isinstance(color, dict) and color.get("field"):
        f = str(color["field"])
        color_card = int(getattr(data_profile, "cardinality", {}).get(f, 0) or 0)

    # time sort missing heuristic
    time_sort_missing = False
    x = enc.get("x")
    if isinstance(x, dict) and str(x.get("type", "")).lower() in ("temporal", "t"):
        if "sort" not in x:
            time_sort_missing = True

    # dual axis/multi y in vega-lite: tricky; detect layering/resolve
    has_dual_axis = False
    multiple_y = False
    if "layer" in spec and isinstance(spec["layer"], list):
        # if multiple layers have y fields -> multi y (often dual axis)
        y_fields = set()
        for layer in spec["layer"]:
            le = (layer or {}).get("encoding", {}) or {}
            y = le.get("y")
            if isinstance(y, dict) and y.get("field"):
                y_fields.add(str(y["field"]))
        if len(y_fields) >= 2:
            multiple_y = True
        resolve = spec.get("resolve")
        if isinstance(resolve, dict):
            sy = resolve.get("scale", {}).get("y")
            if sy == "independent":
                has_dual_axis = True

    return {
        "title": title,
        "mark": mark,
        "encoding": enc,
        "value_axis": value_axis,
        "value_axis_zero": value_axis_zero if value_axis_zero is not None else True,  # assume OK if unspecified
        "color_cardinality": color_card,
        "time_sort_missing": time_sort_missing,
        "has_dual_axis": has_dual_axis,
        "multiple_y": multiple_y,
    }


def parse_plotly(fig: Dict[str, Any], data_profile: Any) -> Dict[str, Any]:
    # Very common structure: fig["data"] list + fig["layout"]
    data = fig.get("data") or []
    layout = fig.get("layout") or {}

    title = layout.get("title")
    if isinstance(title, dict):
        title = title.get("text")

    # determine "mark" approximation from trace types
    # mapping: bar, scatter (mode=lines->line), pie, heatmap
    mark = "unknown"
    trace_types = []
    for tr in data:
        t = (tr or {}).get("type", "scatter")
        trace_types.append(str(t).lower())
    if "pie" in trace_types:
        mark = "pie"
    elif "bar" in trace_types:
        mark = "bar"
    elif "heatmap" in trace_types:
        mark = "heatmap"
    else:
        # scatter: detect lines
        if any((tr.get("mode", "") or "").find("lines") >= 0 for tr in data if isinstance(tr, dict)):
            mark = "line"
        else:
            mark = "point"

    # Build pseudo "encoding" compatible enough for rules
    # Try to infer x/y fields from first trace (if it uses column refs, it might be arrays already)
    enc: Dict[str, Any] = {}

    def _infer_axis(axis_name: str) -> Optional[Dict[str, Any]]:
        ax = layout.get(f"{axis_name}axis") or {}
        # Plotly doesn't preserve "field" name if arrays used
        title2 = ax.get("title")
        if isinstance(title2, dict):
            title2 = title2.get("text")
        return {"axis": {"title": title2}} if title2 else {}

    enc["x"] = _infer_axis("x") or {}
    enc["y"] = _infer_axis("y") or {}

    # zero baseline detection: layout.yaxis.range or autorange
    value_axis_zero = True
    yaxis = layout.get("yaxis") or {}
    if isinstance(yaxis, dict):
        rng = yaxis.get("range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo = rng[0]
            if isinstance(lo, (int, float)) and lo > 0 and mark == "bar":
                value_axis_zero = False

        # log scale
        if str(yaxis.get("type", "")).lower() == "log":
            # rules_engine checks scale.type=log in vega; here we store flag
            enc["y"]["scale"] = {"type": "log"}

    # Color categories: in Plotly it's often legend groups / name; we approximate using data_profile if possible
    # If a trace uses "name" categories, count unique names
    names = set()
    for tr in data:
        if isinstance(tr, dict) and tr.get("name"):
            names.add(str(tr["name"]))
    color_card = len(names)

    # dual axis: yaxis and yaxis2 present
    has_dual_axis = "yaxis2" in layout

    return {
        "title": title,
        "mark": mark,
        "encoding": enc,  # not exact but enough for some readability rules
        "value_axis": "y",
        "value_axis_zero": value_axis_zero,
        "color_cardinality": color_card,
        "time_sort_missing": False,  # hard w/o field info
        "has_dual_axis": has_dual_axis,
        "multiple_y": has_dual_axis,
    }


def detect_spec_type(obj: Dict[str, Any]) -> str:
    # naive detection
    if "$schema" in obj and "vega-lite" in str(obj["$schema"]).lower():
        return "vegalite"
    if "data" in obj and "layout" in obj and "config" in obj:
        # could still be vega; but treat as vegalite-ish
        return "vegalite"
    if "data" in obj and "layout" in obj and isinstance(obj.get("data"), list):
        return "plotly"
    if "data" in obj and "encoding" in obj:
        return "vegalite"
    return "unknown"
