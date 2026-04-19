from instance_loader import load_instance
from week_level.context_builder import build_context
from week_level.arms.coverage_first import generate_schedule_coverage_first

instance = load_instance(
    dataset_root="Dataset/datasets_json",
    dataset_name="n030w4",
    history_idx=0,
    week_indices=[0,1,2,3]
)

context, labels = build_context(
    instance.scenario,
    instance.initial_history,
    instance.weeks[0],
    week_idx=0,
    total_weeks=4
)

assignments, uncovered = generate_schedule_coverage_first(
        scenario=instance.scenario,
        history=instance.initial_history,
        week_data=instance.weeks[0],
    )

print("\nContext:")
for name, value in zip(labels, context):
    print(f"{name}: {value:.4f}")

print(f"Number of assignments created: {len(assignments)}") # assignmenet: {"nurseId", "day", "shiftType", "skill"}
print(f"Number of uncovered demand rows: {len(uncovered)}") # no. of rows that didn't meet minimum coverage

print("\nFirst 10 assignments:")
for row in assignments[:10]:
    print(row)

print("\nFirst 10 uncovered rows:")
for row in uncovered[:10]:
    print(row)
