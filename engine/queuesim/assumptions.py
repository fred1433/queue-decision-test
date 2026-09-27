"""Numbers the decision code needs that no log contains. Each one is set by us and shown as such."""

# Two dials on the same lead by two different centers within this many minutes count as an overlap.
COLLISION_WINDOW_MIN = 20

# Rollout bar: quotes per contacted lead under the change must stay above 95% of the current rate
# (lower bound of the interval), so that extra contacts are not contacts that never quote.
NONINFERIORITY = 0.95

# A decision is reviewed this many weeks after it ships.
REVIEW_WEEKS = 4

# Planner defaults (conditional on assumptions, see power.py).
TARGET_RELATIVE_EFFECT = 0.20
HORIZON_WEEKS = 104
