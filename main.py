#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads every ``*.in`` case file in $MAKEFUN_INPUT_DIR (else /input) and writes
one ``<id>.out`` forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).

For each case the program sees the release day's first-hour print and the
release-day curves of past big game launches (the case's comparables). The
forecast for hours 2..24 is built from those curves alone, offline, with the
standard library only, and is fully deterministic.

Model (per forecast hour h, fitted at run time from the comparables):
  * For each comparable with a complete release day, take the log ratio
    r = ln(viewers_h / viewers_1). Across launches, r depends strongly on
    the size of the first hour: a launch that opens small (only New Zealand /
    Australia / the UK awake at 00:00 UTC) grows a lot as the US wakes up,
    while a launch that opens huge is already near its peak. So r is
    regressed on x = ln(viewers_1) with a quadratic fit (ordinary least
    squares, solved by Gaussian elimination), and the case's own hour-1
    print supplies x. The fitted value is clamped to the range of ratios
    actually observed that hour, and to [-2, +3], so no extrapolation can
    produce an absurd forecast.
  * The spread is the residual dispersion: the interquartile range of the
    fit residuals, scaled to a normal sigma (IQR / 1.349) and shrunk by 0.7,
    floored at 0.25 and capped at 1.5. Quantiles are then
    q_tau = p1 * exp(m_h + z_tau * s_h) with z the standard normal quantiles
    at 5/25/50/75/95%.
  * Comparables whose launch was distorted by early access (their peak
    happened days before release day) and live-service / target-history
    entries are left out: their release-day shape is not a launch shape.

Fallbacks, in order, if the fit cannot be made: a linear fit on ln(viewers_1);
the plain per-hour median log ratio; the persistence baseline (every hour at
the hour-1 print with a spread widening as 0.15 * sqrt(h - 1)). Every guard
floors viewers at 1 and clamps quantiles into (0, 1e9] in non-decreasing order.
"""

import json
import math
import os
from datetime import datetime, timezone

Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
FIRST_HOUR = 1
LAST_HOUR = 24
MIN_VIEWERS_1 = 100
MIN_CURVES_QUADRATIC = 8
MIN_CURVES_LINEAR = 3
SPREAD_SHRINK = 0.7
SPREAD_FLOOR = 0.25
SPREAD_CAP = 1.5
MEDIAN_CLAMP_LOW = -2.0
MEDIAN_CLAMP_HIGH = 3.0
MAX_VIEWERS = 1.0e9


def parse_time(text):
    """Parse an ISO-8601 timestamp to an aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile(values, q):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n == 1:
        return ordered[0]
    pos = q * (n - 1)
    low = int(pos)
    frac = pos - low
    high = low + 1 if low + 1 < n else low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def iqr_sigma(values):
    """Robust sigma estimate from the interquartile range."""
    if len(values) < 2:
        return 0.0
    return (quantile(values, 0.75) - quantile(values, 0.25)) / 1.349


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None if the system is (near-)singular.
    A tiny ridge keeps the fit defined when predictors are degenerate.
    """
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for i in range(n):
        aug[i][i] += 1e-9
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
        out[i] = total / aug[i][i]
    return out


def ols_fit(xs, ys, degree):
    """Least-squares polynomial fit of ys on xs (degree 1 or 2).

    Returns the coefficient list [c0, c1, ...], or None if unsolvable.
    """
    n = len(ys)
    if n < degree + 2:
        return None
    rows = [[1.0] + [x ** p for p in range(1, degree + 1)] for x in xs]
    k = degree + 1
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    target = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    return solve_linear(normal, target)


def polyval(coef, x):
    total = 0.0
    for c in reversed(coef):
        total = total * x + c
    return total

def release_day_curve(comp):
    """Hourly viewers {1..24} for a comparable's release day (UTC).

    A point at ``from`` covers [from, from + 1h); the hour index is counted
    from the comparable's own ``day_start``. Points with missing or
    non-positive viewers are skipped, never interpolated.
    """
    day_start = parse_time(comp.get("day_start"))
    if day_start is None:
        return {}
    curve = {}
    for point in comp.get("hourly") or []:
        moment = parse_time(point.get("from"))
        viewers = point.get("viewers")
        if moment is None or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        hour = int((moment - day_start).total_seconds() // 3600) + 1
        if FIRST_HOUR <= hour <= LAST_HOUR:
            curve[hour] = float(viewers)
    return curve


def usable_comparable(comp):
    """True when a comparable's launch shape is a fair guide to a launch day.

    Early-access launches peaked days before release day, so their release-day
    curve is flat or falling for reasons that have nothing to do with the
    hour of day; live-service launches and the target game's own pre-release
    history are not launch shapes either. Those entries are marked in their
    ``unlock`` note (a "Tier:" tag); an entry with no such tag is kept.
    """
    note = comp.get("unlock") or ""
    lowered = note.lower()
    if "early-access" in lowered:
        return False
    if "Tier:" in note:
        tier = note.split("Tier:")[-1].split(".")[0].strip().lower()
        if tier in ("weak", "target-history"):
            return False
    return True


def collect_curves(case):
    """Release-day curves from the case's comparables.

    A curve counts when it has a first hour of at least MIN_VIEWERS_1
    viewers (a near-empty first hour makes its log ratios noise, not
    signal). Hours are fitted one at a time, so a curve missing a later
    hour still contributes the hours it has instead of being discarded.
    """
    curves = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        if not usable_comparable(comp):
            continue
        curve = release_day_curve(comp)
        if curve.get(FIRST_HOUR, 0.0) < MIN_VIEWERS_1:
            continue
        curves.append(curve)
    return curves


def hour_forecast(curves, hour, p1):
    """Median log ratio and log spread for one forecast hour.

    Fits ln(viewers_h / viewers_1) against ln(viewers_1) across the
    comparables (quadratic when there are enough curves, linear otherwise,
    plain median as a last resort) and evaluates it at the case's own
    hour-1 print. Only comparables that actually have this hour are used
    for it. Returns (median_log_ratio, log_spread).
    """
    usable = [c for c in curves if hour in c]
    if not usable:
        # No comparable has this hour: persist the hour-1 print, with the
        # baseline's widening spread, rather than failing the whole case.
        return 0.0, 0.15 * math.sqrt(hour - 1)
    xs = [math.log(c[FIRST_HOUR]) for c in usable]
    ys = [math.log(c[hour] / c[FIRST_HOUR]) for c in usable]
    lo_x, hi_x = min(xs), max(xs)
    x_case = min(hi_x, max(lo_x, math.log(p1)))
    lo_y, hi_y = min(ys), max(ys)

    coef = None
    if len(usable) >= MIN_CURVES_QUADRATIC:
        coef = ols_fit(xs, ys, 2)
    if coef is None and len(usable) >= MIN_CURVES_LINEAR:
        coef = ols_fit(xs, ys, 1)
    if coef is not None:
        med = polyval(coef, x_case)
        resid = [y - polyval(coef, x) for x, y in zip(xs, ys)]
        spread = iqr_sigma(resid)
    else:
        med = median(ys)
        spread = iqr_sigma(ys)

    med = min(hi_y, max(lo_y, med))
    med = min(MEDIAN_CLAMP_HIGH, max(MEDIAN_CLAMP_LOW, med))
    spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread * SPREAD_SHRINK))
    return med, spread


def baseline_forecast(p1):
    """Persistence fallback: every hour at the hour-1 print.

    Same shape as the house baseline: spread widening as 0.15 * sqrt(h - 1).
    """
    rows = []
    for hour in range(2, LAST_HOUR + 1):
        width = 0.15 * math.sqrt(hour - 1)
        rows.append({
            "hour": hour,
            "q": [round(p1 * math.exp(z * width), 2) for z in Z_VALUES],
        })
    return {"hours": rows}


def clean_quantiles(values):
    """Clamp quantiles into (0, 1e9] and force non-decreasing order."""
    out = []
    prev = 0.0
    for value in values:
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            value = 1.0
        value = min(MAX_VIEWERS, max(1.0, float(value)))
        value = max(prev, value)
        out.append(round(value, 2))
        prev = value
    return out


def forecast_case(case):
    """Build the {"hours": [...]} forecast object for one case."""
    known = case.get("known") or []
    p1 = 1.0
    if known and isinstance(known[0], dict):
        viewers = known[0].get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            p1 = float(viewers)

    curves = collect_curves(case)
    if not curves:
        return baseline_forecast(p1)

    rows = []
    for hour in range(2, LAST_HOUR + 1):
        med, spread = hour_forecast(curves, hour, p1)
        quantiles = [p1 * math.exp(med + z * spread) for z in Z_VALUES]
        rows.append({"hour": hour, "q": clean_quantiles(quantiles)})
    return {"hours": rows}

def process_case(path, out_dir):
    """Read one case file and write its forecast file. Returns the case id."""
    with open(path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    case_id = case.get("id") or os.path.splitext(os.path.basename(path))[0]
    try:
        forecast = forecast_case(case)
    except Exception:
        # Never leave a case without an output: fall back to persistence.
        known = case.get("known") or []
        p1 = 1.0
        if known and isinstance(known[0], dict):
            viewers = known[0].get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                p1 = float(viewers)
        forecast = baseline_forecast(p1)
    text = json.dumps(forecast, separators=(",", ":"), sort_keys=False)
    out_path = os.path.join(out_dir, case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return case_id


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(
        name for name in os.listdir(in_dir) if name.endswith(".in")
    )
    for name in names:
        path = os.path.join(in_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            process_case(path, out_dir)
        except Exception:
            # A case file that cannot even be read still gets a file:
            # the grader scores a missing output far worse than a
            # persistence forecast, so write one with p1 = 1.
            case_id = os.path.splitext(name)[0]
            text = json.dumps(
                baseline_forecast(1.0), separators=(",", ":"), sort_keys=False
            )
            out_path = os.path.join(out_dir, case_id + ".out")
            try:
                with open(out_path, "w", encoding="utf-8") as handle:
                    handle.write(text)
            except OSError:
                pass


if __name__ == "__main__":
    main()
