import numpy as np

DAYS_KEYS = [
    "requirementOnMonday",
    "requirementOnTuesday",
    "requirementOnWednesday",
    "requirementOnThursday",
    "requirementOnFriday",
    "requirementOnSaturday",
    "requirementOnSunday"
]

def build_context(scenario, history, week_data, week_idx, total_weeks):
    nurse_hist = history["nurseHistory"]

    # nurse-level
    consec_work = [n["numberOfConsecutiveWorkingDays"] for n in nurse_hist]
    consec_off = [n["numberOfConsecutiveDaysOff"] for n in nurse_hist]
    assignments = [n["numberOfAssignments"] for n in nurse_hist]

    # aggregate
    features = []
    labels = []

    features.append(np.mean(consec_work))
    labels.append("mean_consec_work_days")

    features.append(np.max(consec_work))
    labels.append("max_consec_work_days")

    features.append(np.mean(consec_off))
    labels.append("mean_consec_off_days")

    features.append(np.min(consec_off))
    labels.append("min_consec_off_days")

    features.append(np.mean(assignments))
    labels.append("mean_assignments")

    # week-level
    requirements = week_data["requirements"]

    total_min_demand = 0

    for r in requirements:
        for day in DAYS_KEYS:
            total_min_demand += r[day]["minimum"]

    features.append(total_min_demand)
    labels.append("total_min_demand")

    # reaquest count
    features.append(len(week_data["shiftOffRequests"]))
    labels.append("num_requests")

    # week position
    features.append(week_idx / total_weeks)
    labels.append("week_position")

    return np.array(features, dtype=float), labels