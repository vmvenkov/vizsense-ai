def rule_pie_too_many_categories(df, data_profile, chart_features):
    issues = []
    if chart_features.mark in ["arc", "pie"]:
        # Vega-Lite pie обикновено е mark: "arc"
        color_field = chart_features.color_categories_field
        if color_field and data_profile.cardinality.get(color_field, 0) > 6:
            issues.append({
                "id": "PIE_TOO_MANY_CATS",
                "message": f"Pie/arc с много категории ({data_profile.cardinality[color_field]}). Bar chart е по-четим.",
                "penalty": 20,
                "suggestion": "Използвай bar chart + сортиране + top-k + Other."
            })
    return issues

def rule_bar_y_should_start_at_zero(df, data_profile, chart_features):
    issues = []
    if chart_features.mark in ["bar"]:
        # ако знаем, че zero=False → penalty
        if chart_features.y_zero is False:
            issues.append({
                "id": "BAR_NONZERO_BASELINE",
                "message": "Bar chart с Y-скала без нулева база може да изкриви сравненията.",
                "penalty": 15,
                "suggestion": "Сложи scale={'zero': True} за Y (ако е подходящо)."
            })
    return issues

def rule_too_many_colors(df, data_profile, chart_features):
    issues = []
    color_field = chart_features.color_categories_field
    if color_field:
        n = data_profile.cardinality.get(color_field, 0)
        if n > 12:
            issues.append({
                "id": "TOO_MANY_COLOR_CATEGORIES",
                "message": f"Твърде много категории в color ({n}) → легендата и цветовете стават шум.",
                "penalty": 10,
                "suggestion": "Top-k категории + Other, или facet/small multiples."
            })
    return issues

ALL_RULES = [
    rule_pie_too_many_categories,
    rule_bar_y_should_start_at_zero,
    rule_too_many_colors,
]
