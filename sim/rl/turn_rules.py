"""Stage 4, fixed 2026-11-04 after baseline and before training (evidence 198).
Tolerance fractions are relative to the commanded speed / yaw rate.
"""
TURN_CRITERIA = {
    1.: dict(yaw=.5, v_tol=.15, yaw_tol=.25, t90=4.8, radius=3.2),
    2.: dict(yaw=1., v_tol=.20, yaw_tol=.35, t90=3.5, radius=3.7),
    4.: dict(yaw=.5, v_tol=.15, yaw_tol=.35, t90=6., radius=14.),
    6.: dict(yaw=.5, v_tol=.15, yaw_tol=.35, t90=6., radius=21.),
}
