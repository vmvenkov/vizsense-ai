from __future__ import annotations

from typing import Any, Dict, List, Optional


def _deepcopy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    # json roundtrip deep copy without importing copy (safe for dict/list primitives)
    import json
    return json.loads(json.dumps(d))


def _get_mark(spec: Dict[str, Any]) -> str:
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    return str(mark).lower() if mark else "unknown"


def _get_encoding(spec: Dict[str, Any]) -> Dict[str, Any]:
    enc = spec.get("encoding")
    return enc if isinstance(enc, dict) else {}


def _ensure_scale_zero_on_quant_axis(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = _deepcopy_dict(spec)
    enc = _get_encoding(out)

    def set_zero(channel: str) -> bool:
        e = enc.get(channel)
        if not isinstance(e, dict):
            return False
        t = str(e.get("type", "")).lower()
        if t not in ("quantitative", "q"):
            return False
        scale = e.get("scale")
        if not isinstance(scale, dict):
            scale = {}
        scale["zero"] = True
        e["scale"] = scale
        enc[channel] = e
        return True

    # Try y then x
    changed = set_zero("y") or set_zero("x")
    if changed:
        out["encoding"] = enc
    return out


def _convert_pie_to_bar(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a simple pie/arc Vega-Lite spec to a sorted bar chart.
    Heuristic:
      - category: encoding.color.field (nominal)
      - value: encoding.theta.field (quantitative)
    Keeps transform as-is.
    """
    out = _deepcopy_dict(spec)
    enc = _get_encoding(out)

    theta = enc.get("theta") if isinstance(enc.get("theta"), dict) else None
    color = enc.get("color") if isinstance(enc.get("color"), dict) else None

    if not theta or not color:
        return out

    cat_field = color.get("field")
    val_field = theta.get("field")
    if not cat_field or not val_field:
        return out

    # Preserve aggregate if present in theta
    val_agg = theta.get("aggregate")

    # Build new encoding: x categorical, y quantitative (sorted)
    new_enc: Dict[str, Any] = {}

    new_enc["x"] = {
        "field": str(cat_field),
        "type": "nominal",
        "sort": "-y",
        "axis": {"labelLimit": 200},
    }

    y_enc: Dict[str, Any] = {"field": str(val_field), "type": "quantitative", "scale": {"zero": True}}
    if val_agg:
        y_enc["aggregate"] = val_agg
    # If original theta had "stack" / other stuff, ignore in MVP
    new_enc["y"] = y_enc

    # carry tooltip if present; otherwise make a sensible one
    if "tooltip" in enc:
        new_enc["tooltip"] = enc["tooltip"]
    else:
        new_enc["tooltip"] = [
            {"field": str(cat_field), "type": "nominal"},
            {"field": str(val_field), "type": "quantitative"},
        ]

    out["mark"] = "bar"
    out["encoding"] = new_enc

    # Remove theta-specific config if present
    out.pop("view", None)
    return out


def _add_time_sort(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    For line/area charts: ensure x encoding has sort ascending (heuristic).
    Vega-Lite supports sort in encoding; for temporal, 'ascending' is ok.
    """
    out = _deepcopy_dict(spec)
    enc = _get_encoding(out)
    x = enc.get("x")
    if isinstance(x, dict):
        # only if temporal-ish or unknown but field exists
        if x.get("field") and "sort" not in x:
            x["sort"] = "ascending"
            enc["x"] = x
            out["encoding"] = enc
    return out


def _topk_color_other(spec: Dict[str, Any], k: int = 10) -> Dict[str, Any]:
    """
    Reduce color categories using top-k + Other.
    This is a best-effort Vega-Lite transform injection.
    Works best when:
      - there is encoding.color.field = category field
      - there is a quantitative measure on y/x/theta (possibly aggregated via transform already)

    Strategy:
      - identify category field from color
      - identify value field from y/x/theta (quant)
      - add window rank over value desc grouped by nothing (after aggregate)
      - calculate new field __cat__ = (rank<=k ? category : 'Other')
      - re-aggregate by __cat__ (sum)
      - set color.field to __cat__
    """
    out = _deepcopy_dict(spec)
    enc = _get_encoding(out)
    color = enc.get("color")
    if not isinstance(color, dict) or not color.get("field"):
        return out

    cat = str(color["field"])

    # find value encoding
    val_field: Optional[str] = None
    val_channel: Optional[str] = None
    for ch in ("y", "x", "theta", "size"):
        e = enc.get(ch)
        if isinstance(e, dict) and str(e.get("type", "")).lower() in ("quantitative", "q") and e.get("field"):
            val_field = str(e["field"])
            val_channel = ch
            break
    if not val_field:
        return out

    # Inject transforms
    transforms = out.get("transform")
    if not isinstance(transforms, list):
        transforms = []

    # Rank by value desc
    transforms.append({
        "window": [{"op": "rank", "as": "__rank__"}],
        "sort": [{"field": val_field, "order": "descending"}]
    })
    # Create grouped category
    transforms.append({
        "calculate": f"datum.__rank__ <= {int(k)} ? datum['{cat}'] : 'Other'",
        "as": "__cat__"
    })

    # Re-aggregate by __cat__ (sum value)
    transforms.append({
        "aggregate": [{"op": "sum", "field": val_field, "as": "__val__"}],
        "groupby": ["__cat__"]
    })

    out["transform"] = transforms

    # Update encoding to use __cat__ as category, __val__ as value
    # Keep orientation based on original val_channel if possible
    # If original had y quantitative, keep y quantitative; else use y quantitative by default
    # and x nominal.
    out_enc: Dict[str, Any] = {}
    out_enc["x"] = {"field": "__cat__", "type": "nominal", "sort": "-y", "axis": {"labelLimit": 200}}
    out_enc["y"] = {"field": "__val__", "type": "quantitative", "scale": {"zero": True}}

    # Tooltip
    out_enc["tooltip"] = [
        {"field": "__cat__", "type": "nominal"},
        {"field": "__val__", "type": "quantitative"},
    ]

    # If mark is line/area, top-k color other usually not desired; but keep it anyway as a fix suggestion
    out["encoding"] = out_enc
    out["mark"] = "bar" if _get_mark(out) in ("arc", "pie") else out.get("mark", "bar")

    return out


def apply_fix_actions_vegalite(spec: Dict[str, Any], fixes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply fix payloads from rules_engine.FixAction.payload (type-based).
    """
    out = _deepcopy_dict(spec)

    for fx in fixes:
        fx_type = fx.get("type")

        if fx_type == "set_zero_baseline":
            out = _ensure_scale_zero_on_quant_axis(out)

        elif fx_type == "convert_pie_to_bar":
            out = _convert_pie_to_bar(out)

        elif fx_type == "add_time_sort":
            out = _add_time_sort(out)

        elif fx_type == "topk_color_other":
            k = int(fx.get("k", 10) or 10)
            out = _topk_color_other(out, k=k)

        # unknown fix -> ignore safely

    return out


def make_spec_renderable_with_df(spec: Dict[str, Any], df, max_rows: int = 5000) -> Dict[str, Any]:
    """
    Many uploaded Vega-Lite specs use data: {"name":"table"} or external refs.
    For preview in Streamlit, we inject inline data values.

    If df is too large, we sample to keep UI fast.
    """
    import pandas as pd

    out = _deepcopy_dict(spec)
    if df is None or len(df) == 0:
        return out

    if isinstance(df, pd.DataFrame) and len(df) > max_rows:
        df2 = df.sample(n=max_rows, random_state=42)
    else:
        df2 = df

    records = df2.to_dict(orient="records")

    # Inject at top level
    out["data"] = {"values": records}
    return out
