# Synthetic Maintenance Dataset Design

## Purpose

This project does not yet have access to historical client maintenance records. The first dataset is therefore **synthetic** and is designed for educational analysis only. It must not be represented as real customer, technician, building, or asset data.

The generator simulates a denormalized CMMS-style work-order dataset that can support data cleaning, exploratory analysis, visualization, and later machine-learning experiments for work-order priority prediction.

## Planned Scale

- 10 simulated properties
- approximately 750 simulated assets
- 3 years of work-order history (2023-01-01 through 2025-12-31)
- approximately 8,000 base work orders
- a small number of intentional duplicate records added after generation
- fixed random seed (`42`) for reproducibility

## Analysis Unit

Each row represents one maintenance work order enriched with asset and maintenance-history features that would commonly be available from a CMMS or asset-management system.

## Fields

| Field | Type | Description |
| --- | --- | --- |
| `work_order_id` | string | Synthetic unique work-order identifier |
| `property_id` | string | Synthetic property identifier |
| `asset_id` | string | Synthetic asset identifier |
| `created_date` | date | Work-order creation date |
| `asset_type` | category | Asset category such as HVAC, plumbing, or electrical |
| `asset_age_years` | numeric | Approximate asset age when the work order was created |
| `asset_condition` | category | Good, Fair, or Poor |
| `asset_criticality` | category | Standard, Important, or Critical |
| `maintenance_type` | category | Corrective, Preventive, or Inspection |
| `issue_type` | category | Maintenance issue associated with the asset category |
| `previous_failures_12m` | integer | Simulated failures for the same asset during the prior 12 months |
| `days_since_last_service` | numeric | Days since the asset was last serviced |
| `pm_overdue_days` | numeric | Estimated number of days preventive maintenance is overdue |
| `occupancy_impact` | category | None, Low, Medium, or High |
| `safety_related` | boolean | Whether the work order involves a simulated safety concern |
| `estimated_repair_cost` | numeric | Simulated estimated repair cost in U.S. dollars |
| `resolution_hours` | numeric | Simulated hours until completion |
| `priority` | category | Target label: Low, Medium, High, or Emergency |

## Asset Types

The simulated portfolio contains a weighted mix of common facilities assets:

- HVAC
- Electrical
- Plumbing
- Elevator
- Fire/Life Safety
- Lighting
- Pump
- Appliance
- Access Control
- Building Envelope

The mix is intentionally uneven because real facilities portfolios rarely contain equal numbers of every asset type.

## Domain-Informed Relationships

The data is not generated as independent random columns. The generator introduces probabilistic relationships intended to resemble plausible maintenance behavior.

### Asset age and condition

Older assets are more likely to be in `Fair` or `Poor` condition than newer assets, but random variation remains so age does not determine condition perfectly.

### Failure history

Assets that are older or in worse condition are more likely to have a higher value for `previous_failures_12m`.

### Preventive-maintenance compliance

Long periods since service increase the probability that `pm_overdue_days` is positive. Overdue preventive maintenance contributes modestly to the probability of a more urgent priority, but it does not determine priority by itself.

### Criticality and safety

Fire/life-safety, elevator, electrical, and selected access-control issues are more likely to be safety-related or operationally critical. Safety-related work orders receive a strong increase in urgent-priority probability.

### Occupancy impact

Issues that materially affect building occupants are more likely to be categorized as High or Emergency than issues with little or no occupancy impact.

### Repair cost

Estimated repair cost depends on asset type, age, condition, failure history, and random variation. High cost alone must not automatically imply a high priority.

### Resolution time

Resolution time reflects issue complexity and operational urgency. This field occurs after work-order creation and would be considered a potential **data-leakage feature** if the future model is intended to predict priority at intake time.

## Priority Generation

`priority` is the eventual prediction target with four classes:

- Low
- Medium
- High
- Emergency

The generator creates a latent urgency score using several features, including:

- safety-related status
- occupancy impact
- asset criticality
- asset condition
- previous failures
- preventive-maintenance lateness
- maintenance type
- controlled random noise

Score thresholds are selected to produce an intentionally imbalanced target distribution near:

| Priority | Approximate Share |
| --- | ---: |
| Low | 22% |
| Medium | 47% |
| High | 25% |
| Emergency | 6% |

A small percentage of labels are randomly perturbed after scoring. This represents operational inconsistency and prevents priority from being a perfectly deterministic function of the input features.

## Intentional Data-Quality Problems

The raw dataset intentionally contains limited imperfections so that the course cleaning requirements represent meaningful work rather than artificial transformations on already-perfect data.

### Missing values

A small percentage of records will contain missing values in fields such as:

- `asset_condition`
- `days_since_last_service`
- `estimated_repair_cost`

Missingness is kept limited so the dataset remains usable while allowing discussion of imputation and deletion bias.

### Inconsistent categorical formatting

A small percentage of categorical values will use inconsistent capitalization or surrounding whitespace, for example:

- `HVAC`, `hvac`, ` HVAC `
- `High`, `HIGH`, `high`

### Duplicate work orders

A small sample of rows will be duplicated while retaining the same `work_order_id`. This provides a defensible duplicate-removal task.

### Outliers / impossible values

A very small number of records may contain intentionally invalid or extreme values such as:

- negative asset age
- unusually large resolution time
- unusually large estimated repair cost

These records must be investigated rather than automatically deleted without explanation.

## Bias Considerations

Synthetic data can encode the assumptions of its designer. For example, assigning higher safety likelihood to particular asset categories or linking poor condition to higher priority may create patterns that appear statistically strong even though they were intentionally generated.

Cleaning can add another source of bias. If incomplete records occur disproportionately in one priority class or asset category, dropping every incomplete row could change the apparent population distribution. The notebook should inspect missingness before deciding whether to remove or impute values.

## Future Real-Data Validation

Before using a model derived from this workflow in a real maintenance environment, the following would need to be validated against actual operational records:

- feature definitions and availability at prediction time
- true priority-class distribution
- maintenance terminology and category mappings
- missing-data patterns
- class imbalance
- asset-type mix
- failure-history quality
- technician or dispatcher differences in priority assignment
- fairness and operational bias across buildings, locations, and asset groups
- model calibration and error costs

## Machine-Learning Leakage Boundary

For a future model that predicts work-order priority **at the moment a request is created**, features known only after completion must not be used as predictors. In particular, `resolution_hours` should be excluded from such a model, and any cost field should be checked to ensure it represents an estimate available at intake rather than final realized repair cost.

## Reproducibility

The generator will use a fixed random seed defined in `src/config.py`. Running the generator with the same configuration and compatible package versions should reproduce the same synthetic dataset.
