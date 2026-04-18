# src — Scheduling Strategies

Five scheduling strategies for the INRC-2 nurse rostering problem, plus
shared utilities. All strategies take the same inputs and return the same
output format, so a bandit can swap between them as arms.

---

## Shared input / output

**Every strategy function takes:**
- `scenario` — the full scenario JSON (nurses, skills, forbidden successions)
- `history` — nurse history JSON for the current week (last shift worked, assignment counts)
- `week_data` — weekly demand JSON (requirements, shift-off requests)

**Every strategy returns:**
- `assignments` — list of `{nurseId, day, shiftType, skill}` dicts
- `uncovered` — list of slots where minimum staffing could not be met, with shortage count

**Hard constraints respected by all strategies:**
- Nurse must have the required skill
- A nurse can only work one shift per day
- Forbidden shift-type successions (e.g. night → early next day) are blocked using previous day's assignment or `lastAssignedShiftType` from history

---

## Files

### `loader.py`
Loads INRC-2 dataset files from disk.

- Scans a dataset directory for `Sc-*.json` (scenario), `H0-*.json` (history), `WD-*.json` (weekly demand)
- `load_instance(dataset_root, dataset_name, history_idx, week_indices)` — main entry point, returns an `INRCInstance` dataclass
- Validates that exactly one scenario file exists and at least one history and week file

---

### `context_builder.py`
Builds a numeric feature vector from the current scheduling state.

- Per-nurse metrics: consecutive days worked, consecutive days off, total assignments
- Aggregate metrics: mean/max consecutive work days, mean/min consecutive days off across all nurses
- Week-level: total minimum staffing demand, total shift-off requests
- Temporal: current week as a fraction of total weeks (0.0 → 1.0)
- Returns a NumPy array + descriptive labels — used as the context input for a bandit

---

### `coverage_first.py`
**Greedy. Fills minimum staffing requirements slot by slot.**

- Extracts all (day, shiftType, skill, minimum) slots from `week_data`
- For each slot, finds feasible nurses and sorts them by penalty score:
  - +100 if nurse has a general shift-off request on that day
  - +150 if nurse has a request for that specific shift type
- Assigns the lowest-penalty nurses first until the slot's minimum is met
- Logs a shortage if minimum cannot be reached
- Never looks ahead — each slot is decided independently

---

### `preference_first.py`
**Minimises nurse dissatisfaction. Protects nurses with no requests.**

- Sorts slots so those with fewer shift-off request conflicts are filled first — giving nurses in high-conflict slots the best chance of being free when their slot is reached
- Within each slot, assigns nurses with zero penalty (no requests) before those with requests — same scoring as coverage_first but the slot order is different
- Still fills to minimum; logs shortages the same way

---

### `fairness_first.py`
**Spreads workload evenly. Least-worked nurse goes first.**

- For each slot, fetches each candidate's total shifts worked: `numberOfAssignments` from history + shifts already assigned this week
- Sorts candidates so the nurse with the fewest total shifts is assigned first
- Does not score shift-off request penalties — fairness takes priority over preferences
- Ties broken alphabetically by nurse ID for determinism

---

### `max_coverage.py`
**Fills each slot to its maximum staffing level, not just minimum.**

- Reads both `minimum` and `maximum` from each slot's requirements
- Assigns nurses until `maximum` is reached (rather than stopping at `minimum`)
- Uses the same penalty-based candidate ordering as coverage_first
- Logs a shortage only if even the `minimum` cannot be met
- Useful when the schedule needs a buffer against absences or late constraint changes

---

### `random_baseline.py`
**Randomly selects from feasible candidates. Lower-bound benchmark.**

- Shuffles the feasible candidate list randomly before assigning
- Accepts an optional `seed` argument for reproducible runs
- No scoring, no preference, no fairness — pure random within hard constraints
- Any learned strategy should outperform this over time

---

## How these connect to a bandit

Each strategy is one **arm**. At the start of each week the bandit:
1. Calls `context_builder.py` to get the feature vector for the current state
2. Picks an arm (strategy) based on its learned policy
3. Runs that strategy to produce the week's assignments
4. Receives a reward (e.g. coverage met − total penalty − shortage count)
5. Updates its policy based on the reward

Over many episodes the bandit learns which strategy works best in which
context — e.g. fairness_first may work better early in a long rotation,
while max_coverage may work better when the roster is thin.
