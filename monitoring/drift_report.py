import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# Load reference (training) data
reference_data = pd.read_csv("data/reference.csv")

# Load current (production) data
current_data = pd.read_csv("data/current.csv")

# Create a data drift report
report = Report(metrics=[
    DataDriftPreset()
])

# Run the report
report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save the report as HTML
report.save_html("monitoring/drift_report.html")

print("Drift report generated successfully.")