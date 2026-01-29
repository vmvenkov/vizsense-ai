from dataclasses import dataclass

@dataclass
class ScoreResult:
    score: int
    breakdown: dict
    issues: list

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))

def score_chart(data_profile, chart_features, issues):
    base = 85
    breakdown = {"base": base}

    penalty = 0
    for iss in issues:
        penalty += iss.get("penalty", 0)

    breakdown["penalty_total"] = penalty
    final = clamp(base - penalty)

    return ScoreResult(score=final, breakdown=breakdown, issues=issues)
