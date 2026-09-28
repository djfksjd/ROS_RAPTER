"""Debounced footstep metrics."""

import numpy as np


def steps(force, slip_speed, foot_x, dt, weight,
          load_frac=0.2, unload_frac=0.05, hold=0.03):
    """Find touchdowns, completed stances, and step metrics for one foot.

    Slip samples without contact (NaN) add no slip. Only stances ended by a debounced liftoff are kept.
    """
    force = np.asarray(force)
    slip_speed = np.asarray(slip_speed)
    foot_x = np.asarray(foot_x)
    if any(a.ndim != 1 for a in (force, slip_speed, foot_x)):
        raise ValueError("inputs must be 1-D")
    if len(force) != len(slip_speed) or len(force) != len(foot_x):
        raise ValueError("inputs must have equal length")
    if dt <= 0 or hold < 0:
        raise ValueError("dt must be positive and hold nonnegative")

    loaded = bool(len(force) and force[0] >= load_frac * weight)
    unloaded_from = None if loaded else 0
    run_start = None
    touchdown = None
    touchdowns = []
    stances = []

    def add_stance(start, end):
        stances.append({
            "start": start * dt,
            "end": end * dt,
            "slip_m": float(np.nansum(slip_speed[start:end]) * dt),
            "x_touchdown": float(foot_x[start]),
        })

    for i, value in enumerate(force):
        qualifies = (value < unload_frac * weight if loaded
                     else value >= load_frac * weight)
        if not qualifies:
            run_start = None
            continue
        if run_start is None:
            run_start = i
        if (i - run_start + 1) * dt + 1e-12 < hold:
            continue

        if loaded:
            if touchdown is not None:
                add_stance(touchdown, run_start)
                touchdown = None
            unloaded_from = run_start
        else:
            touchdown = run_start
            if unloaded_from is not None and (run_start - unloaded_from) * dt + 1e-12 >= hold:
                touchdowns.append(run_start)
            else:  # the preceding unloaded period was not observed for `hold`
                touchdown = None
        loaded = not loaded
        run_start = None

    # A stance still open at the end is incomplete and is not reported.

    lengths = [
        float(foot_x[b] - foot_x[a])
        for a, b in zip(touchdowns, touchdowns[1:])
    ]
    slips = [stance["slip_m"] for stance in stances]
    return {
        "touchdown_times": [i * dt for i in touchdowns],
        "stances": stances,
        "step_lengths": lengths,
        "median_step": float(np.median(lengths)) if lengths else None,
        "median_slip": float(np.median(slips)) if slips else None,
        "count": len(touchdowns),
    }


def summarize(left, right, cycles):
    """Summarize two feet across rocking cycles."""
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    feet = {"left": left, "right": right}
    return {
        "touchdowns_per_foot_per_cycle": {
            name: foot["count"] / cycles for name, foot in feet.items()
        },
        "median_step": {
            name: foot["median_step"] for name, foot in feet.items()
        },
        "median_slip": {
            name: foot["median_slip"] for name, foot in feet.items()
        },
        "slip_to_step": {
            name: (
                foot["median_slip"] / abs(foot["median_step"])
                if foot["median_step"] is not None
                and foot["median_step"] != 0
                and foot["median_slip"] is not None
                else None
            )
            for name, foot in feet.items()
        },
    }
