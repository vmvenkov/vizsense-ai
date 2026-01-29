from .rules import ALL_RULES

def run_rules(df, data_profile, chart_features):
    issues = []
    for r in ALL_RULES:
        issues.extend(r(df, data_profile, chart_features))
    return issues
