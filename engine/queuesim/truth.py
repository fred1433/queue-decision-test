"""The hidden truth of the simulated operation.

The simulator reads this file. The decision engine never does (tests/test_isolation.py enforces it):
the engine sees only the logs an operation like this one keeps. Every number here is either
assumed; the weekly scale is the target the funnel was tuned to land near.

The calibration week: 3,200 leads, 1,100 quotes, 40 issued policies, 3 collected;
ad spend 164,000 MXN; call center cost 234,000 MXN; three call centers with different hourly
rates dialing the same leads; an AI voice agent taking overflow.
"""

CALIBRATION_WEEK = {
    "leads": 3200,
    "quotes": 1100,
    "issued": 40,
    "collected": 3,
    "ad_spend_mxn": 164_000,
    "call_center_cost_mxn": 234_000,
}

# Operating calendar: Monday to Saturday, dialing slots 08:00 to 20:00 (last slot ends 21:00).
OPEN_DAYS = 6
SLOTS = list(range(8, 21))

# ---------------------------------------------------------------- lead sources
# Leads per week and cost per lead. Meta + Google spend = 164,000 MXN exactly.
SOURCES = {
    #            share   CPL (MXN)  contact  quote|contact  issue|quote  collect  premium median (MXN/yr)
    "meta":     dict(share=0.55, cpl=55.0, contact=1.00, quote=0.95, issue=0.80, collect=0.80, premium=7800),
    "google":   dict(share=0.28, cpl=75.0, contact=1.10, quote=1.10, issue=1.35, collect=1.05, premium=9500),
    "database": dict(share=0.17, cpl=0.0,  contact=0.80, quote=0.90, issue=1.10, collect=1.40, premium=11000),
}
PREMIUM_SIGMA = 0.45  # lognormal spread of annual premium

# Hour of arrival of ad leads (0..23), relative weights. Database leads are loaded at 08:00.
ARRIVAL_WEIGHTS = [
    0.6, 0.4, 0.3, 0.2, 0.2, 0.3, 0.6, 1.0, 1.6, 2.2, 2.6, 2.9,
    3.3, 3.4, 2.8, 2.4, 2.4, 2.7, 3.2, 3.8, 4.0, 3.4, 2.2, 1.2,
]

# ---------------------------------------------------------------- contact process
BASE_CONTACT = 0.285          # per dial, fresh lead, hour multiplier 1, attempt 1
HOUR_CONTACT = {               # the true value of the hour: people answer after work
    8: 0.74, 9: 0.84, 10: 0.88, 11: 0.90, 12: 0.93, 13: 0.95, 14: 0.90,
    15: 0.90, 16: 0.96, 17: 1.06, 18: 1.22, 19: 1.27, 20: 1.16,
}
AGE_FLOOR = 0.55               # contact multiplier for a stale lead
AGE_TAU_MIN = 90.0             # minutes: speed-to-lead decay constant
ATTEMPT_FATIGUE = 0.92         # per extra attempt
CONTACTABILITY_SHAPE = 1.6     # gamma shape of a lead's latent reachability (mean 1)
COLLISION_PROB = 0.12          # a second center dials the same lead within 20 minutes
COLLISION_BURN = 0.55          # after a collision, the lead answers less (hidden truth)

# ---------------------------------------------------------------- handlers
CENTERS = {
    #        MXN per agent hour (set by us within the 234,000 total), quote, issue, capture on call
    "A": dict(rate=165.0, quote=1.05, issue=1.20, capture=0.36, agents=14),
    "B": dict(rate=135.0, quote=1.00, issue=1.00, capture=0.20, agents=16),
    "C": dict(rate=110.0, quote=0.93, issue=0.88, capture=0.09, agents=18),
}
CENTER_SPLIT = {"A": 0.30, "B": 0.33, "C": 0.37}
AI = dict(contact=0.90, quote=0.78, issue=0.50, capture=0.0, cost_per_min=3.0)
TENURE_QUOTE_GAIN = 0.10       # agents with 6+ months quote 10% more often
SCRIPT_EFFECT = 1.00           # script B vs A: planted null, no effect at all

BASE_QUOTE = 0.68              # per contact
BASE_ISSUE = 0.033             # per quote
INTENT_SIGMA = 0.55            # latent intent, lognormal; drives quote, issue, capture and collection

# Collection of the first payment within 30 days of issue.
COLLECT_IF_CAPTURED = 0.175     # payment taken on the call (card or direct debit)
COLLECT_IF_LINK = 0.022        # payment link sent after the call
CAPTURE_INTENT_POWER = 0.6     # committed buyers give their card more often (unlogged confounder)
COLLECT_INTENT_POWER = 0.5     # and also pay more often anyway

# Status quo staffing, agents on the phone per slot, all centers together (Mon-Sat).
STAFFING = {8: 38, 9: 31, 10: 31, 11: 30, 12: 29, 13: 22, 14: 24,
            15: 23, 16: 21, 17: 17, 18: 9, 19: 7, 20: 5}
DIAL_SHARE = 0.55              # share of an agent hour available for new dialing (rest: closing work)
OCCUPANCY_CAP = 0.88
HANDLE_MIN = {"no_answer": 1.4, "contact": 6.0, "quote": 22.0}
AI_HANDLE_MIN = {"no_answer": 0.8, "contact": 3.5, "quote": 9.0}

# Status quo dialing policy ("intuition"): up to 10 attempts; retry gaps in minutes.
MAX_ATTEMPTS = 10
RETRY_GAPS_MIN = [25, 120, 240, 1440, 1440, 2880, 2880, 2880, 4320]
# The platform's priority score: Google first, then database, then Meta. Humans take the top of the queue,
# the AI agent takes what is left of the new leads.
PRIORITY = {"google": 0, "database": 1, "meta": 2}
