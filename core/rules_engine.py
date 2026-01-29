from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

import math


Criterion = Literal["Appropriateness", "Readability", "Integrity", "Accessibility"]
SourceType = Literal["vegalite", "plotly", "image", "unknown"]
Confidence = Literal["High", "Medium", "Low"]

CRITERIA: List[Criterion] = ["Appropriateness", "Readability", "Integrity", "Accessibility"]


@dataclass
class Issue:
    id: str
    criterion: Criterion
    severity: Literal["info", "warning", "error"]
    penalty: int
    message: str
    suggestion: str


@dataclass
class FixAction:
    id: str
    description: str
    # A generic payload you can use later to apply fixes
    payload: Dict[str, Any]


@dataclass
class RuleContext:
    # from data profiler
    data_profile: Any  # your DataProfile
    # parsed chart information
    source_type: SourceType
    chart_meta: Dict[str, Any]  # normalized across specs/images
    # optional raw spec for fixers
    raw_spec: Optional[Dict[str, Any]] = None
    # for image mode
    image_meta: Optional[Dict[str, Any]] = None


@dataclass
class RuleResult:
    issues: List[Issue]
    fixes: List[FixAction]


@dataclass
class ScoreResult:
    total: int
    by_criterion: Dict[Criterion, int]
    confidence: Confidence
    issues: List[Issue]
    fixes: List[FixAction]


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def calc_confidence(source_type: SourceType) -> Confidence:
    if source_type in ("vegalite", "plotly"):
        return "High"
    if source_type == "image":
        return "Medium"
    return "Low"


def score_from_issues(issues: List[Issue], confidence: Confidence) -> Tuple[int, Dict[Criterion, int]]:
    # Base: 25 each criterion
    by: Dict[Criterion, int] = {c: 25 for c in CRITERIA}

    # In image mode we slightly soften penalties on Appropriateness/Integrity
    soften = 1.0
    soften_app_int = 0.7 if confidence == "Medium" else 1.0

    for iss in issues:
        pen = iss.penalty
        if confidence == "Medium" and iss.criterion in ("Appropriateness", "Integrity"):
            pen = int(round(pen * soften_app_int))
        pen = int(round(pen * soften))
        by[iss.criterion] = max(0, by[iss.criterion] - pen)

    total = sum(by.values())
    return clamp_int(total, 0, 100), by


# ---------------------------
# Helpers for chart_meta fields
# ---------------------------
def _get(meta: Dict[str, Any], key: str, default=None):
    return meta.get(key, default)


def _is_mark(meta: Dict[str, Any], *marks: str) -> bool:
    m = str(_get(meta, "mark", "unknown")).lower()
    return any(m == x.lower() for x in marks)


def _encoding(meta: Dict[str, Any], channel: str) -> Optional[Dict[str, Any]]:
    enc = _get(meta, "encoding", {}) or {}
    v = enc.get(channel)
    return v if isinstance(v, dict) else None


def _field(enc: Optional[Dict[str, Any]]) -> Optional[str]:
    if not enc:
        return None
    f = enc.get("field")
    return str(f) if f else None


def _type(enc: Optional[Dict[str, Any]]) -> Optional[str]:
    if not enc:
        return None
    t = enc.get("type")
    return str(t) if t else None


def _aggregate(enc: Optional[Dict[str, Any]]) -> Optional[str]:
    if not enc:
        return None
    a = enc.get("aggregate")
    return str(a) if a else None


def _scale(enc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not enc:
        return {}
    s = enc.get("scale")
    return s if isinstance(s, dict) else {}


def _axis(enc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not enc:
        return {}
    a = enc.get("axis")
    return a if isinstance(a, dict) else {}


# ---------------------------
# Rules
# ---------------------------
def run_all_rules(ctx: RuleContext) -> RuleResult:
    # rule packs
    issues: List[Issue] = []
    fixes: List[FixAction] = []

    # universal (works on any source using data_profile)
    issues += rule_high_missing_rate(ctx)
    issues += rule_high_cardinality_categories(ctx)

    # spec-only rules
    if ctx.source_type in ("vegalite", "plotly"):
        issues += rule_missing_title(ctx)
        issues += rule_axis_titles_missing(ctx)
        issues += rule_pie_too_many_categories(ctx)
        issues += rule_bar_baseline_zero(ctx)
        issues += rule_too_many_color_categories(ctx)
        issues += rule_line_without_time(ctx)
        issues += rule_time_as_category(ctx)
        issues += rule_scatter_overplotting(ctx)
        issues += rule_unaggregated_bar(ctx)
        issues += rule_wrong_aggregate_semantics(ctx)
        issues += rule_log_scale_invalid(ctx)
        issues += rule_truncated_domain_suspicious(ctx)
        issues += rule_sort_time_missing(ctx)
        issues += rule_legend_overload(ctx)
        issues += rule_stacked_too_many_series(ctx)
        issues += rule_dual_axis_or_multi_y(ctx)
        issues += rule_heatmap_not_matrix(ctx)
        issues += rule_histogram_not_numeric(ctx)
        issues += rule_binning_mismatch(ctx)

        # Fix suggestions (spec-only)
        fixes += fix_bar_zero_baseline(ctx)
        fixes += fix_convert_pie_to_bar(ctx)
        fixes += fix_add_time_sort(ctx)
        fixes += fix_limit_topk_color(ctx)

    # image-only rules
    if ctx.source_type == "image":
        issues += rule_image_low_resolution(ctx)
        issues += rule_image_low_contrast(ctx)
        issues += rule_image_too_many_colors(ctx)
        issues += rule_image_clutter(ctx)

    # De-duplicate issues by id (keep highest penalty)
    best: Dict[str, Issue] = {}
    for iss in issues:
        if iss.id not in best or iss.penalty > best[iss.id].penalty:
            best[iss.id] = iss

    return RuleResult(list(best.values()), fixes)


# ===========================
# Universal / Data-driven rules
# ===========================
def rule_high_missing_rate(ctx: RuleContext) -> List[Issue]:
    p = ctx.data_profile
    mr = float(getattr(p, "missing_rate", 0.0) or 0.0)
    if mr >= 0.20:
        return [Issue(
            id="DATA_MISSING_HIGH",
            criterion="Integrity",
            severity="warning",
            penalty=6,
            message=f"Липсващи стойности са високи (~{mr*100:.0f}%). Това може да изкриви визуалните изводи.",
            suggestion="Филтрирай/импутирай липсващите стойности и покажи колко са премахнати."
        )]
    if mr >= 0.05:
        return [Issue(
            id="DATA_MISSING_MED",
            criterion="Integrity",
            severity="info",
            penalty=3,
            message=f"Има забележим дял липсващи стойности (~{mr*100:.0f}%).",
            suggestion="Провери дали липсващите стойности влияят на агрегациите и графиката."
        )]
    return []


def rule_high_cardinality_categories(ctx: RuleContext) -> List[Issue]:
    p = ctx.data_profile
    cat_cols = list(getattr(p, "categorical_cols", []) or [])
    card = dict(getattr(p, "cardinality", {}) or {})
    if not cat_cols:
        return []
    max_col = None
    max_n = 0
    for c in cat_cols:
        n = int(card.get(c, 0) or 0)
        if n > max_n:
            max_n = n
            max_col = c
    if max_col and max_n >= 30:
        return [Issue(
            id="DATA_HIGH_CARDINALITY",
            criterion="Readability",
            severity="warning",
            penalty=6,
            message=f"Категорийна колона '{max_col}' има много уникални стойности ({max_n}) → риск от претрупана графика.",
            suggestion="Използвай top-k + Other, филтър, фасети или агрегиране."
        )]
    return []


# ===========================
# Spec-based rules (Vega-Lite / Plotly normalized meta)
# ===========================
def rule_missing_title(ctx: RuleContext) -> List[Issue]:
    title = _get(ctx.chart_meta, "title")
    if not title:
        return [Issue(
            id="VIZ_NO_TITLE",
            criterion="Readability",
            severity="info",
            penalty=3,
            message="Липсва заглавие на графиката.",
            suggestion="Добави ясна цел/въпрос в заглавието (напр. “Revenue by Channel, Jan–Feb 2025”)."
        )]
    return []


def rule_axis_titles_missing(ctx: RuleContext) -> List[Issue]:
    enc = ctx.chart_meta.get("encoding", {}) or {}
    missing = 0
    for ch in ("x", "y"):
        e = enc.get(ch)
        if isinstance(e, dict):
            axis = e.get("axis")
            if isinstance(axis, dict) and axis.get("title") in ("", None) and e.get("field"):
                missing += 1
            # if axis not provided, it's okay; vega-lite auto titles
    if missing:
        return [Issue(
            id="VIZ_AXIS_TITLES_MISSING",
            criterion="Readability",
            severity="info",
            penalty=2 * missing,
            message="Липсват/са празни заглавия на оси за някои канали.",
            suggestion="Добави axis.title за по-ясна интерпретация."
        )]
    return []


def rule_pie_too_many_categories(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "arc", "pie"):
        return []
    color = _encoding(ctx.chart_meta, "color")
    f = _field(color)
    if not f:
        return []
    p = ctx.data_profile
    n = int(getattr(p, "cardinality", {}).get(f, 0) or 0)
    if n > 6:
        return [
            Issue(
                id="PIE_TOO_MANY_CATS",
                criterion="Appropriateness",
                severity="warning",
                penalty=12,
                message=f"Pie/arc с много категории ({n}) затруднява сравненията.",
                suggestion="Използвай bar chart (сортирано) и/или top-k + Other."
            ),
            Issue(
                id="PIE_READABILITY",
                criterion="Readability",
                severity="warning",
                penalty=6,
                message="При много сектори легендата/цветовете стават трудни за следене.",
                suggestion="Намали категориите или използвай facet/small multiples."
            )
        ]
    return []


def rule_bar_baseline_zero(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "bar"):
        return []
    y = _encoding(ctx.chart_meta, "y")
    x = _encoding(ctx.chart_meta, "x")
    # whichever is quantitative axis for value
    val_enc = y if (_type(y) in ("quantitative", "Q")) else x if (_type(x) in ("quantitative", "Q")) else None
    if not val_enc:
        return []
    scale = _scale(val_enc)
    if scale.get("zero") is False:
        return [Issue(
            id="BAR_NONZERO_BASELINE",
            criterion="Integrity",
            severity="error",
            penalty=12,
            message="Bar chart с изключена нулева база (scale.zero=false) може да изкриви визуалното сравнение.",
            suggestion="Задай scale.zero=true за стойностната ос, освен ако няма силна причина."
        )]
    return []


def rule_too_many_color_categories(ctx: RuleContext) -> List[Issue]:
    color = _encoding(ctx.chart_meta, "color")
    f = _field(color)
    if not f:
        return []
    p = ctx.data_profile
    n = int(getattr(p, "cardinality", {}).get(f, 0) or 0)
    if n > 12:
        return [Issue(
            id="TOO_MANY_COLOR_CATEGORIES",
            criterion="Accessibility",
            severity="warning",
            penalty=10,
            message=f"Твърде много категории в color ({n}) → претоварена легенда и трудно различими цветове.",
            suggestion="Top-k + Other, facet по категории или интерактивен филтър."
        )]
    return []


def rule_line_without_time(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "line", "area"):
        return []
    x = _encoding(ctx.chart_meta, "x")
    if not x:
        return []
    t = _type(x)
    # expect temporal
    if t not in ("temporal", "T"):
        return [Issue(
            id="LINE_WITHOUT_TIME",
            criterion="Appropriateness",
            severity="warning",
            penalty=8,
            message="Line/area графика без времева ос (temporal) често внушава тенденция там, където няма.",
            suggestion="Ако X е категориен → използвай bar; ако са две числови → scatter."
        )]
    return []


def rule_time_as_category(ctx: RuleContext) -> List[Issue]:
    # temporal column encoded as nominal/ordinal => readability/appropriateness issue
    x = _encoding(ctx.chart_meta, "x")
    f = _field(x)
    if not f:
        return []
    p = ctx.data_profile
    col_types = dict(getattr(p, "col_types", {}) or {})
    if col_types.get(f) == "datetime" and _type(x) in ("nominal", "ordinal", "N", "O"):
        return [Issue(
            id="TIME_AS_CATEGORY",
            criterion="Appropriateness",
            severity="warning",
            penalty=6,
            message="Времева колона е кодирана като категорийна (nominal/ordinal).",
            suggestion="Задай temporal тип (':T') и използвай line/area или подходяща агрегация."
        )]
    return []


def rule_scatter_overplotting(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "point", "circle", "square"):
        return []
    p = ctx.data_profile
    n_rows = int(getattr(p, "n_rows", 0) or 0)
    if n_rows > 20000:
        return [Issue(
            id="SCATTER_OVERPLOT_HEAVY",
            criterion="Readability",
            severity="warning",
            penalty=10,
            message=f"Много точки ({n_rows}) → силно наслагване (overplotting).",
            suggestion="Използвай binning/hexbin, агрегация, sampling и/или opacity."
        )]
    if n_rows > 5000:
        return [Issue(
            id="SCATTER_OVERPLOT",
            criterion="Readability",
            severity="info",
            penalty=6,
            message=f"Доста точки ({n_rows}) → вероятно наслагване.",
            suggestion="Добави opacity (напр. 0.2–0.5) и/или агрегация."
        )]
    return []


def rule_unaggregated_bar(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "bar"):
        return []
    # If x is categorical and y is quantitative and no aggregate, likely many rows -> should aggregate
    x = _encoding(ctx.chart_meta, "x")
    y = _encoding(ctx.chart_meta, "y")
    if not x or not y:
        return []
    x_t = _type(x)
    y_t = _type(y)
    if x_t in ("nominal", "ordinal", "N", "O") and y_t in ("quantitative", "Q"):
        if not _aggregate(y):
            # if dataset has many rows, more likely wrong
            p = ctx.data_profile
            if int(getattr(p, "n_rows", 0) or 0) > 50:
                return [Issue(
                    id="BAR_NO_AGGREGATE",
                    criterion="Appropriateness",
                    severity="warning",
                    penalty=8,
                    message="Bar chart по категорийна ос без агрегиране на стойностите може да е подвеждащо/шумно.",
                    suggestion="Добави aggregate (sum/mean/count) според смисъла или предварително групирай данните."
                )]
    return []


def rule_wrong_aggregate_semantics(ctx: RuleContext) -> List[Issue]:
    # Heuristic: if field name suggests totals (revenue/sales/units) and aggregate is mean -> warning
    encs = ctx.chart_meta.get("encoding", {}) or {}
    suspects = ("revenue", "sales", "amount", "units", "count", "total", "profit")
    issues: List[Issue] = []

    for ch in ("x", "y", "theta", "size"):
        e = encs.get(ch)
        if not isinstance(e, dict):
            continue
        f = e.get("field")
        agg = e.get("aggregate")
        if not f or not agg:
            continue
        fname = str(f).lower()
        if any(s in fname for s in suspects) and str(agg).lower() in ("mean", "average", "avg"):
            issues.append(Issue(
                id=f"AGG_SEMANTICS_{ch.upper()}",
                criterion="Integrity",
                severity="info",
                penalty=4,
                message=f"Полето '{f}' изглежда като “тотал/сума”, но е агрегирано с mean.",
                suggestion="Провери дали не трябва sum вместо mean (особено за приходи/бройки)."
            ))
    return issues


def rule_log_scale_invalid(ctx: RuleContext) -> List[Issue]:
    # If scale.type=log but data might contain 0/neg (we can't be sure; use profile heuristics)
    encs = ctx.chart_meta.get("encoding", {}) or {}
    p = ctx.data_profile
    numeric_cols = set(getattr(p, "numeric_cols", []) or [])
    issues: List[Issue] = []

    for ch in ("x", "y"):
        e = encs.get(ch)
        if not isinstance(e, dict):
            continue
        f = e.get("field")
        sc = e.get("scale")
        if not f or not isinstance(sc, dict):
            continue
        if str(sc.get("type", "")).lower() == "log" and f in numeric_cols:
            issues.append(Issue(
                id=f"LOG_SCALE_{ch.upper()}",
                criterion="Integrity",
                severity="warning",
                penalty=6,
                message=f"Използвана е log скала на {ch}-оста. Ако има 0/отрицателни стойности, резултатът може да е некоректен.",
                suggestion="Увери се, че всички стойности са > 0, или използвай symlog/linear и ясно обозначи скалата."
            ))
    return issues


def rule_truncated_domain_suspicious(ctx: RuleContext) -> List[Issue]:
    # If scale.domain is narrow & bar chart => possible exaggeration
    if not _is_mark(ctx.chart_meta, "bar"):
        return []
    y = _encoding(ctx.chart_meta, "y")
    x = _encoding(ctx.chart_meta, "x")
    val_enc = y if (_type(y) in ("quantitative", "Q")) else x if (_type(x) in ("quantitative", "Q")) else None
    if not val_enc:
        return []
    sc = _scale(val_enc)
    domain = sc.get("domain")
    if isinstance(domain, list) and len(domain) == 2:
        lo, hi = domain[0], domain[1]
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and hi > lo:
            # If domain doesn't include 0 and range is small relative to hi -> suspicious
            if lo > 0 and (hi - lo) / max(1e-9, abs(hi)) < 0.25:
                return [Issue(
                    id="TRUNCATED_DOMAIN_SUSPECT",
                    criterion="Integrity",
                    severity="warning",
                    penalty=6,
                    message="Стойностната ос използва тесен domain без 0 → може да преувеличи разликите.",
                    suggestion="Върни включване на 0 (особено за bar), или обясни защо domain е ограничен."
                )]
    return []


def rule_sort_time_missing(ctx: RuleContext) -> List[Issue]:
    # For line chart with temporal x, ensure sorting by x
    if not _is_mark(ctx.chart_meta, "line", "area"):
        return []
    x = _encoding(ctx.chart_meta, "x")
    if not x or _type(x) not in ("temporal", "T"):
        return []
    # Heuristic: if encoding.sort exists and is not by x, ok; otherwise warn
    if "sort" not in x:
        return [Issue(
            id="TIME_SORT_MISSING",
            criterion="Integrity",
            severity="info",
            penalty=3,
            message="Line/area по време без изрично сортиране може да даде “назъбена” линия при несортирани данни.",
            suggestion="Сортирай по времевата колона или добави sort по X."
        )]
    return []


def rule_legend_overload(ctx: RuleContext) -> List[Issue]:
    # Approx: if color field has high cardinality, legend overload
    color = _encoding(ctx.chart_meta, "color")
    f = _field(color)
    if not f:
        return []
    p = ctx.data_profile
    n = int(getattr(p, "cardinality", {}).get(f, 0) or 0)
    if n > 15:
        return [Issue(
            id="LEGEND_OVERLOAD",
            criterion="Readability",
            severity="warning",
            penalty=6,
            message=f"Легендата вероятно е претоварена ({n} елемента).",
            suggestion="Намали категориите, използвай facet, или интерактивно highlight."
        )]
    return []


def rule_stacked_too_many_series(ctx: RuleContext) -> List[Issue]:
    # Vega-Lite stacking: if mark bar/area and color has many categories => hard to compare
    if not _is_mark(ctx.chart_meta, "bar", "area"):
        return []
    # heuristics: stack defaults for area; for bar often stacked if color is present
    color = _encoding(ctx.chart_meta, "color")
    f = _field(color)
    if not f:
        return []
    p = ctx.data_profile
    n = int(getattr(p, "cardinality", {}).get(f, 0) or 0)
    if n >= 6:
        return [Issue(
            id="STACKED_MANY_SERIES",
            criterion="Appropriateness",
            severity="warning",
            penalty=6,
            message=f"Stacked визуализация с много серии ({n}) затруднява сравненията между категории.",
            suggestion="Използвай facet/small multiples или grouped bar вместо stacked."
        )]
    return []


def rule_dual_axis_or_multi_y(ctx: RuleContext) -> List[Issue]:
    # Works best if parser populates meta flags
    if _get(ctx.chart_meta, "has_dual_axis") is True:
        return [Issue(
            id="DUAL_AXIS",
            criterion="Integrity",
            severity="warning",
            penalty=10,
            message="Dual-axis визуализация може да бъде подвеждаща и трудна за интерпретация.",
            suggestion="Използвай small multiples или нормализация + една ос."
        )]
    if _get(ctx.chart_meta, "multiple_y") is True:
        return [Issue(
            id="MULTI_Y",
            criterion="Appropriateness",
            severity="warning",
            penalty=6,
            message="Има повече от една Y-метрика в една графика → риск от претрупване/неяснота.",
            suggestion="Раздели в отделни графики или използвай facet."
        )]
    return []


def rule_heatmap_not_matrix(ctx: RuleContext) -> List[Issue]:
    if not _is_mark(ctx.chart_meta, "rect", "heatmap"):
        return []
    x = _encoding(ctx.chart_meta, "x")
    y = _encoding(ctx.chart_meta, "y")
    c = _encoding(ctx.chart_meta, "color")
    if not x or not y or not c:
        return [Issue(
            id="HEATMAP_INCOMPLETE",
            criterion="Appropriateness",
            severity="warning",
            penalty=6,
            message="Heatmap/rect без ясни X/Y и color кодирания.",
            suggestion="Heatmap обикновено изисква две оси (категории/време) + количествена стойност в color."
        )]
    # If x or y is quantitative, might be ok, but typical heatmap uses discrete bins/categories
    if _type(c) not in ("quantitative", "Q"):
        return [Issue(
            id="HEATMAP_COLOR_NOT_QUANT",
            criterion="Appropriateness",
            severity="warning",
            penalty=6,
            message="Heatmap color не е количествен (quantitative).",
            suggestion="Използвай количествена метрика за color (напр. count/sum/mean)."
        )]
    return []


def rule_histogram_not_numeric(ctx: RuleContext) -> List[Issue]:
    # histogram is typically bar with bin=True on a quantitative axis
    if not _is_mark(ctx.chart_meta, "bar"):
        return []
    x = _encoding(ctx.chart_meta, "x")
    y = _encoding(ctx.chart_meta, "y")

    # detect histogram pattern: x has bin and is quantitative, y is count()
    if x and isinstance(x.get("bin"), (bool, dict)):
        if _type(x) not in ("quantitative", "Q"):
            return [Issue(
                id="HIST_BIN_NOT_NUMERIC",
                criterion="Appropriateness",
                severity="warning",
                penalty=8,
                message="Има binning върху не-числова ос → хистограма трябва да е върху числова променлива.",
                suggestion="Избери числова колона за histogram, или махни bin и използвай bar count by category."
            )]
    # If y is count and x is nominal, it's fine (count by category)
    return []


def rule_binning_mismatch(ctx: RuleContext) -> List[Issue]:
    # binning on very low-cardinality numeric could be unnecessary; or missing binning for many unique numeric in bar
    x = _encoding(ctx.chart_meta, "x")
    if not x:
        return []
    f = _field(x)
    if not f:
        return []
    p = ctx.data_profile
    col_types = dict(getattr(p, "col_types", {}) or {})
    card = dict(getattr(p, "cardinality", {}) or {})
    if col_types.get(f) == "numeric":
        n = int(card.get(f, 0) or 0)
        has_bin = isinstance(x.get("bin"), (bool, dict))
        if n > 50 and _is_mark(ctx.chart_meta, "bar") and not has_bin and _type(x) in ("quantitative", "Q"):
            return [Issue(
                id="MISSING_BINNING_FOR_MANY_UNIQUE",
                criterion="Readability",
                severity="info",
                penalty=4,
                message=f"Числова колона '{f}' има много уникални стойности ({n}). Bar по сурови стойности може да е шум.",
                suggestion="Използвай binning (histogram) или агрегация."
            )]
    return []


# ===========================
# Image rules
# ===========================
def rule_image_low_resolution(ctx: RuleContext) -> List[Issue]:
    im = ctx.image_meta or {}
    w = int(im.get("width", 0) or 0)
    h = int(im.get("height", 0) or 0)
    if w and h and (w < 800 or h < 450):
        return [Issue(
            id="IMG_LOW_RES",
            criterion="Readability",
            severity="warning",
            penalty=8,
            message=f"Ниска резолюция ({w}×{h}) → дребен текст и ниска четимост.",
            suggestion="Експортирай с по-висока резолюция (напр. 1600×900+) или като vector (SVG/PDF)."
        )]
    return []


def rule_image_low_contrast(ctx: RuleContext) -> List[Issue]:
    im = ctx.image_meta or {}
    c = float(im.get("contrast", 1.0) or 1.0)
    if c < 0.10:
        return [Issue(
            id="IMG_LOW_CONTRAST",
            criterion="Accessibility",
            severity="warning",
            penalty=8,
            message="Нисък контраст → текст/оси може да са трудни за четене.",
            suggestion="Увеличи контраста (по-тъмен текст/оси, по-светъл фон), избягвай бледи цветове."
        )]
    return []


def rule_image_too_many_colors(ctx: RuleContext) -> List[Issue]:
    im = ctx.image_meta or {}
    n = int(im.get("n_unique_colors_approx", 0) or 0)
    if n > 25:
        return [Issue(
            id="IMG_MANY_COLORS",
            criterion="Accessibility",
            severity="info",
            penalty=6,
            message="Много различни цветове → вероятен визуален шум / трудна легенда.",
            suggestion="Ограничи палитрата и използвай акцентен цвят само за важните елементи."
        )]
    return []


def rule_image_clutter(ctx: RuleContext) -> List[Issue]:
    im = ctx.image_meta or {}
    ed = float(im.get("edge_density", 0.0) or 0.0)
    if ed > 0.12:
        return [Issue(
            id="IMG_CLUTTER",
            criterion="Readability",
            severity="info",
            penalty=6,
            message="Графиката изглежда претрупана (висока плътност на детайли).",
            suggestion="Намали gridlines, редуцирай етикети, използвай агрегация или small multiples."
        )]
    return []


# ===========================
# Fix actions (spec-only)
# ===========================
def fix_bar_zero_baseline(ctx: RuleContext) -> List[FixAction]:
    if ctx.source_type not in ("vegalite", "plotly"):
        return []
    if not _is_mark(ctx.chart_meta, "bar"):
        return []
    if _get(ctx.chart_meta, "value_axis_zero") is True:
        return []
    # propose fix
    return [FixAction(
        id="FIX_BAR_ZERO_BASELINE",
        description="Задай нулева база (scale.zero=true) за стойностната ос на bar chart.",
        payload={"type": "set_zero_baseline"}
    )]


def fix_convert_pie_to_bar(ctx: RuleContext) -> List[FixAction]:
    if ctx.source_type not in ("vegalite", "plotly"):
        return []
    if not _is_mark(ctx.chart_meta, "arc", "pie"):
        return []
    return [FixAction(
        id="FIX_PIE_TO_BAR",
        description="Преобразувай pie/arc към сортиран bar chart (по-четимо).",
        payload={"type": "convert_pie_to_bar"}
    )]


def fix_add_time_sort(ctx: RuleContext) -> List[FixAction]:
    if ctx.source_type not in ("vegalite", "plotly"):
        return []
    if not _is_mark(ctx.chart_meta, "line", "area"):
        return []
    if _get(ctx.chart_meta, "time_sort_missing") is True:
        return [FixAction(
            id="FIX_ADD_TIME_SORT",
            description="Добави сортиране по времето за line/area графика.",
            payload={"type": "add_time_sort"}
        )]
    return []


def fix_limit_topk_color(ctx: RuleContext) -> List[FixAction]:
    if ctx.source_type not in ("vegalite", "plotly"):
        return []
    if _get(ctx.chart_meta, "color_cardinality", 0) and int(ctx.chart_meta["color_cardinality"]) > 12:
        return [FixAction(
            id="FIX_TOPK_COLOR",
            description="Ограничи категориите в color (top-k + Other) или замени с facet.",
            payload={"type": "topk_color_other", "k": 10}
        )]
    return []


# ===========================
# Public API: analyze
# ===========================
def analyze(ctx: RuleContext) -> ScoreResult:
    confidence = calc_confidence(ctx.source_type)
    rr = run_all_rules(ctx)
    total, by = score_from_issues(rr.issues, confidence)

    return ScoreResult(
        total=total,
        by_criterion=by,
        confidence=confidence,
        issues=sorted(rr.issues, key=lambda x: (-x.penalty, x.criterion, x.id)),
        fixes=rr.fixes
    )
