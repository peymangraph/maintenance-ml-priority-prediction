# Maintenance ML Priority Prediction

A Python data-workflow project that creates, cleans, explores, and visualizes realistic synthetic maintenance work-order data as a foundation for future machine-learning priority prediction.

> **Current phase:** The AI Programming Foundations data-workflow phase is implemented. Model training is intentionally reserved for a later machine-learning phase.

## Project Overview

This project builds a professional, reusable workflow around maintenance work orders and asset-management information. The long-term goal is to predict work-order priority (`Low`, `Medium`, `High`, or `Emergency`) from operational information available at or before request intake.

The current course project focuses on the foundation required before machine learning:

- reproducible synthetic-data generation;
- Pandas data ingestion;
- reusable cleaning functions with docstrings;
- exploratory data analysis (EDA);
- interpreted Matplotlib visualizations;
- bias and data-quality reflection;
- reproducible Python dependencies;
- explicit assumptions and limitations.

## Why Maintenance Data?

Public tutorial datasets such as Titanic are useful for learning, but this repository is designed around a facilities-management use case that can later become a real machine-learning application. The same workflow can eventually support work-order priority classification, predictive maintenance, asset-failure analysis, repair-cost estimation, and intelligent work-order routing.

Because historical customer maintenance records are not yet available for this project, the initial dataset is **synthetic**.

## Synthetic Data Disclosure

The records in this project are simulated and do **not** represent actual customers, employees, technicians, properties, or assets.

The generator creates domain-informed statistical relationships while deliberately introducing controlled data-quality problems such as:

- missing values;
- inconsistent capitalization and whitespace;
- duplicate work-order records;
- impossible negative asset ages;
- injected repair-cost and resolution-time outliers.

These imperfections make the cleaning workflow meaningful. Relationships discovered in the data partly reflect assumptions encoded by the generator and must not be presented as validated evidence about real maintenance organizations.

## Implemented Dataset

The default generator uses a fixed random seed (`42`) and creates:

- **10 simulated properties**;
- approximately **750 simulated assets**;
- **8,000 base work orders** across 2023–2025;
- **8,040 raw rows** after intentional duplicate injection;
- **18 columns**;
- four future target classes: `Low`, `Medium`, `High`, and `Emergency`.

The raw priority mix is intentionally imbalanced, with Medium as the largest group and Emergency as the smallest group. This creates a more useful foundation for later discussion of precision, recall, F1 score, confusion matrices, and class imbalance.

### Main fields

| Category | Fields |
| --- | --- |
| Identifiers | `work_order_id`, `property_id`, `asset_id` |
| Time | `created_date` |
| Asset | `asset_type`, `asset_age_years`, `asset_condition`, `asset_criticality` |
| Maintenance | `maintenance_type`, `issue_type`, `previous_failures_12m`, `days_since_last_service`, `pm_overdue_days` |
| Operational impact | `occupancy_impact`, `safety_related` |
| Cost/outcome | `estimated_repair_cost`, `resolution_hours` |
| Future target | `priority` |

The detailed generation assumptions are documented in `docs/dataset_design.md`.

## Workflow

```text
Synthetic CMMS Data Generator
            |
            v
      Raw CSV Dataset
            |
            v
       Pandas Ingestion
            |
            v
        Data Cleaning
            |
            v
 Exploratory Data Analysis
            |
            v
      Visualizations
            |
            v
     Processed Dataset
            |
            v
 Future ML Feature Engineering
```

The cleaned dataset contains the **8,000 unique base work orders** after duplicate removal and documented handling of missing/invalid values.

## Repository Structure

```text
maintenance-ml-priority-prediction/
|
|-- README.md
|-- SECURITY.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- notebooks/
|   `-- maintenance_data_workflow.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- generate_data.py
|   |-- cleaning.py
|   |-- eda.py
|   `-- visualizations.py
|
|-- docs/
|   `-- dataset_design.md
|
|-- reports/
|   `-- figures/
|
`-- models/
```

The CSV files can be regenerated from the committed Python code. The notebook automatically generates the raw dataset when it is not present.

## Notebook Structure

The primary notebook contains the exact course-oriented sections:

1. **Setup**
2. **Data Ingestion**
3. **Data Cleaning**
4. **Exploratory Data Analysis**
5. **Visualizations**
6. **Summary and Interpretation**

The final submission should be executed from a restarted kernel from top to bottom.

## Reproducing the Project

Clone the repository:

```bash
git clone https://github.com/peymangraph/maintenance-ml-priority-prediction.git
cd maintenance-ml-priority-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
notebooks/maintenance_data_workflow.ipynb
```

Restart the kernel and run all cells from the beginning.

### Requirements File

The pinned package versions in `requirements.txt` were derived from `pip freeze` in the project execution environment. The environment includes Pandas, NumPy, Matplotlib, Jupyter/Notebook, IPython kernel support, and notebook-format support.

## Data Cleaning Functions

`src/cleaning.py` implements four reusable student-defined functions, each with an informative docstring:

### `standardize_categories()`

Normalizes equivalent asset-type and priority labels. For example, `HVAC`, `hvac`, and ` HVAC ` should be treated as one category rather than three separate values.

### `remove_duplicate_work_orders()`

Removes duplicate records using `work_order_id`. Duplicate events would otherwise distort work-order counts, costs, and priority distributions.

### `flag_invalid_values()`

Changes documented impossible or deliberately injected values to missing values before imputation. The thresholds are specific to this synthetic educational dataset and are not intended as universal maintenance rules.

### `impute_missing_values()`

Labels missing asset condition as `Unknown` rather than inventing a Good/Fair/Poor value. Numerical missing values are filled with group medians using asset type and maintenance type, followed by an overall median fallback.

## Cleaning Justification

The project avoids simply deleting every incomplete record because row deletion can change which asset categories or priority classes are represented. Median imputation is used for selected skewed numerical variables because it is less sensitive to extreme values than a mean, but the notebook explicitly acknowledges that imputation still reduces variability.

The raw dataset remains unchanged and cleaning transformations are applied to a separate DataFrame. The processed output is saved separately so the workflow remains auditable.

## Bias Awareness: How Poor Cleaning Can Introduce Bias

Poor data cleaning can change the population represented by a dataset and therefore change the conclusions or models built from it.

For example, imagine that low-priority maintenance requests are more likely to have incomplete documentation because technicians spend less time recording routine work. If every row with a missing field were deleted, low-priority work could be removed at a higher rate than emergency work. The cleaned dataset would then overrepresent serious incidents and a future classifier could learn a distorted view of normal operations.

Imputation can also introduce bias. Replacing every missing repair cost or asset age with one overall average can erase meaningful differences among asset types or maintenance categories. Aggressive outlier removal can be equally harmful because a rare emergency failure may be operationally important rather than erroneous.

Category normalization and duplicate removal also require domain context. Similar-looking labels may represent different equipment classes, and two similar work orders may be repeated failures rather than duplicates. For this project, the duplicate IDs and malformed categories are intentionally generated, so their treatment is known. With real data, those rules would require validation with business owners and data documentation.

## Exploratory Data Analysis

`src/eda.py` contains reusable functions for:

- priority counts and percentages;
- work-order/cost/failure summaries by asset type;
- operational summaries by priority;
- transparent filtering of records with repeated failures and overdue PM.

The notebook also displays descriptive statistics for numerical maintenance fields.

A critical principle throughout the analysis is that **association is not causation**. Because this dataset is synthetic, observed patterns may directly reflect the assumptions used to generate it.

## Visualizations

`src/visualizations.py` provides reusable Matplotlib functions. The notebook currently includes four labeled and interpreted visualizations:

1. work-order distribution by priority;
2. average previous failures by priority;
3. median estimated repair cost by asset type;
4. resolution-time distribution by priority.

Every visualization has a descriptive title, labeled axes, and a written interpretation immediately following the chart.

## Data Leakage Note

`resolution_hours` is useful for retrospective operational analysis but is **not appropriate as a feature for predicting priority at intake** because it is only known after a work order has been resolved.

A future machine-learning phase must distinguish information available at prediction time from post-outcome information. This prevents target leakage and unrealistic model performance.

## Future ML Workflow Reflection

When this project advances from exploratory analysis to supervised machine learning, the target (`priority`) will be separated from candidate predictor features and the data will be divided into training, validation, and test sets.

Preprocessing steps that learn statistics from data must be fitted only on the training set. Categorical fields will need encoding, and numerical fields may require scaling depending on the selected algorithm. A baseline should be established before complex models are introduced.

Because Emergency work orders are less common than Medium work orders, accuracy alone would be insufficient. Appropriate evaluation should include:

- precision;
- recall;
- F1 score;
- per-class metrics;
- confusion matrix;
- possibly class-weighted evaluation.

Candidate later models include logistic regression, decision trees, random forests, and gradient-boosted trees. Their performance should be compared against a simple baseline and interpreted within the maintenance use case rather than choosing a model only because it has the largest score.

## Neural Network Preparation Reflection

A neural network would require additional preparation because the table contains categorical, numerical, Boolean, and potentially future text features.

Categorical values such as asset type, issue type, and maintenance type would need numerical representations such as one-hot encoding or learned embeddings. Numerical values such as asset age, PM overdue days, and previous failure counts would normally be scaled or normalized. Inputs and targets would then be converted to tensors for the chosen deep-learning framework.

The training, validation, and test split would still need to be respected, and imputation/scaling statistics must be learned only from training data. Class imbalance could require class weighting or carefully evaluated sampling strategies.

A neural network should not be assumed to outperform classical models. For a tabular dataset of this size, tree-based or linear models may perform as well or better while remaining easier to interpret.

## Agentic Automation Potential Reflection

The completed data workflow could eventually support an agentic maintenance system. A future agent could:

```text
Receive maintenance request
          |
          v
Identify property / asset
          |
          v
Retrieve maintenance history
          |
          v
Call priority prediction model
          |
          v
Recommend priority
          |
          v
Assist with work-order creation / routing
          |
          v
Monitor status and request missing information
```

The machine-learning model would be one tool used by the agent, not the entire decision-making system.

Human oversight remains essential for life-safety events, emergency response, regulatory decisions, costly replacements, and uncertain predictions. A production version would require real operational validation, role-based access, audit logging, confidence thresholds, privacy controls, exception handling, and clear human-override rules.

## Assumptions and Limitations

The major limitation is that this project uses synthetic rather than observed customer data.

- Relationships reflect generator rules and probabilities, not measured customer behavior.
- Synthetic data can encode the assumptions or biases of its designer.
- Performance on this data cannot demonstrate real-world predictive performance.
- Actual organizations may use very different asset taxonomies and priority definitions.
- Real data may contain seasonality, technician effects, property effects, reporting inconsistencies, and missingness mechanisms not represented here.
- Cleaning thresholds and imputation choices must be reevaluated before use with real records.

Once real operational data becomes available, distributions and relationships should be re-audited and every cleaning and feature-engineering assumption should be validated before training a production model.

## Project Roadmap

- [x] Initialize Python repository structure.
- [x] Create a development branch beyond `main`.
- [x] Document the synthetic maintenance dataset specification.
- [x] Build the reproducible Python data generator.
- [x] Create the primary Jupyter notebook and ingest data with Pandas.
- [x] Implement documented data-cleaning functions.
- [x] Perform exploratory data analysis with reusable functions.
- [x] Add at least three interpreted visualizations.
- [x] Complete notebook summary, assumptions, and limitations.
- [x] Add pinned `requirements.txt` versions derived from `pip freeze`.
- [ ] Perform the final clean-kernel, top-to-bottom execution audit.
- [ ] Complete final rubric review and merge the submission branch.
- [ ] Future phase: train and evaluate priority-classification models.

## Git/GitHub Workflow

Development uses multiple meaningful commits on `feature/project-setup` in addition to `main`, satisfying the course expectation that Git history reflects incremental work rather than one final bulk upload.

The branch is reviewed through a pull request before being merged into the public `main` branch.

## Security

This is a public repository. **Never commit credentials or real customer data.**

Do not commit:

- API keys;
- passwords or tokens;
- `.env` secret files;
- private keys or certificates;
- production database connection strings;
- cloud credentials;
- customer or personally identifiable maintenance data.

See `SECURITY.md` and `.gitignore` for additional safeguards.

## Academic and Ethical Note

This repository is an educational data-workflow and machine-learning foundation project. Synthetic records are explicitly identified as synthetic, and conclusions are presented with appropriate limitations. No synthetic analysis should be represented as evidence collected from real clients or real maintenance operations.
