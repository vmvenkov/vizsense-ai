import altair as alt

def apply_basic_fixes(chart: alt.Chart, spec: dict) -> alt.Chart:
    # за MVP: ако bar и y.scale.zero=False -> направи True
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")

    if mark == "bar":
        enc = spec.get("encoding", {}) or {}
        y = enc.get("y")
        if isinstance(y, dict):
            scale = y.get("scale") or {}
            if isinstance(scale, dict) and scale.get("zero") is False:
                # rebuild chart with corrected y scale
                y2 = y.copy()
                y2["scale"] = dict(scale)
                y2["scale"]["zero"] = True

                new_spec = dict(spec)
                new_enc = dict(enc)
                new_enc["y"] = y2
                new_spec["encoding"] = new_enc
                return alt.Chart.from_dict(new_spec)
    return chart
