# Maintenance ML Priority Prediction

A Python data-workflow project that prepares realistic maintenance and work-order data for future machine-learning models that predict maintenance priority.

> **Current phase:** Data preparation, cleaning, exploratory analysis, visualization, and reproducibility. Model training is intentionally a later phase.

## Project Overview

This project builds a professional, reusable data workflow around maintenance work orders and asset-management information. The long-term goal is to predict work-order priority (`Low`, `Medium`, `High`, or `Emergency`) from operational features such as asset age, maintenance history, condition, prior failures, preventive-maintenance compliance, issue type, occupancy impact, and safety impact.

The current course project focuses on the foundation required before machine learning: generating or ingesting data, cleaning it with reusable Python functions, performing exploratory data analysis (EDA), creating interpretable visualizations, documenting assumptions and limitations, and making the workflow reproducible.

## Why Maintenance Data?

Public tutorial datasets such as Titanic are useful for learning, but this repository is designed around a real facilities-management use case. A maintenance workflow creates a stronger foundation for later work in classification, predictive maintenance, asset-failure forecasting, repair-cost estimation, and intelligent work-order routing.

Because real customer maintenance history is not yet available for this project, the initial dataset will be **synthetic**.

## Synthetic Data Disclosure

The maintenance records used in the first version of this project are simulated and do **not** represent actual customer, employee, property, technician, or asset records.

The synthetic generator is designed to create realistic-looking relationships among maintenance variables while also introducing controlled data-quality problems such as missing values, inconsistent categorical formatting, duplicates, and limited outliers. These imperfections give the cleaning workflow meaningful problems to address.

Synthetic relationships are assumptions created for educational experimentation. Results from this dataset must not be interpreted as validated evidence about real buildings, assets, technicians, or maintenance organizations.

## Project Objective

The immediate objective is to create an ML-ready maintenance dataset through a reproducible Python workflow:

```text
Synthetic Maintenance Data
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
   ML-Ready Dataset
```

A later phase can extend this workflow into supervised machine learning:

```text
ML-Ready Dataset
       |
       v
Feature Engineering
       |
       v
Train / Validation / Test Split
       |
       v
Baseline + ML Models
       |
       v
Model Evaluation
       |
       v
Priority Prediction Service
```

## Planned Dataset

The synthetic dataset is expected to represent multiple properties, hundreds of assets, and several years of work-order history. The initial design targets roughly 8,000 work orders.

Example feature groups include:

| Category | Example Fields |
| --- | --- |
| Work order | `work_order_id`, `created_date`, `issue_type`, `maintenance_type`, `priority` |
| Asset | `asset_id`, `asset_type`, `asset_age_years`, `condition`, `criticality` |
| Maintenance history | `previous_failures_12m`, `days_since_last_service`, `pm_overdue_days` |
| Operational impact | `occupancy_impact`, `safety_related` |
| Outcome | `resolution_hours`, `labor_hours`, `repair_cost`, `repeat_failure_30d` |

The target for the future classification task is:

```text
Low
Medium
High
Emergency
```

The detailed generation rules and assumptions will be documented in `docs/dataset_design.md`.

## Repository Structure

The project is organized as a Python-first data-science repository:

```text
maintenance-ml-priority-prediction/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- raw/
|   |   `-- maintenance_work_orders.csv
|   `-- processed/
|       `-- maintenance_ml.csv
|
|-- notebooks/
|   `-- maintenance_data_workflow.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- generate_data.py
|   |-- preprocess.py
|   `-- config.py
|
|-- docs/
|   `-- dataset_design.md
|
|-- reports/
|   `-- figures/
|
`-- models/
```

Some directories and files above are part of the planned workflow and will be added incrementally as the project issues are completed.

## Notebook Structure

The primary notebook will contain the course-required sections with clear Markdown headings:

1. **Setup**
2. **Data Ingestion**
3. **Data Cleaning**
4. **Exploratory Data Analysis**
5. **Visualizations**
6. **Summary and Interpretation**

The notebook must run from top to bottom without errors.

## Python Environment

The project uses Python and Jupyter. The environment will be kept intentionally small and focused on the libraries actually used by the analysis.

Expected core packages include:

- Python 3.11+
- pandas
- NumPy
- Matplotlib
- Jupyter
- Seaborn, if used in the final notebook
- scikit-learn in the later machine-learning phase

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

Install the exact project dependencies after `requirements.txt` has been generated:

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

Then run the notebook from the first cell through the final cell.

### Requirements File

To meet the project reproducibility requirement, the final environment dependency list will be generated from the working virtual environment with:

```bash
pip freeze > requirements.txt
```

This ensures that the submitted `requirements.txt` records the package versions used to execute the project.

## Data Cleaning Strategy

The raw synthetic dataset will intentionally contain realistic data-quality problems. Cleaning functions will be implemented as reusable Python functions with informative docstrings and will be called from the notebook.

Planned cleaning tasks include:

- standardizing categorical text and whitespace;
- handling missing values using field-appropriate strategies;
- identifying true duplicate work-order records;
- validating impossible or suspicious numerical values;
- preserving rare but legitimate observations, especially high-severity maintenance events.

Each cleaning decision will be explained in the notebook so that the transformation is transparent rather than hidden inside code.

## Bias Awareness: How Poor Data Cleaning Can Introduce Bias

Poor cleaning can change the population represented by a dataset and therefore change the conclusions or models produced from it. For example, suppose low-priority work orders are more likely to have incomplete fields because technicians spend less time documenting routine requests. If every row containing a missing value were simply deleted, low-priority work could be removed at a higher rate than emergency work. The cleaned dataset would then overrepresent serious incidents and could produce misleading statistics or a biased future classifier.

Imputation can also create bias. Replacing all missing repair costs or asset ages with one overall average can erase differences among asset types, buildings, or maintenance categories. Likewise, aggressive outlier removal could incorrectly delete rare emergency failures that are operationally important. In maintenance data, an extreme value is not automatically an error.

Category standardization also requires care. Values such as `HVAC`, `hvac`, and ` HVAC ` can reasonably be normalized to one category, but two labels that look similar may represent genuinely different equipment classes. Duplicate removal has the same risk: two records describing similar failures may be repeated maintenance events rather than accidental duplicate rows. For that reason, cleaning rules should use identifiers and domain context instead of removing records only because their values look unusual.

These examples show why cleaning decisions must be documented and evaluated for their effect on class distribution, asset groups, and other important subpopulations.

## Future ML Workflow Reflection

When this project advances from exploratory analysis to machine learning, several workflow changes will be necessary. The target variable will be separated from candidate features, and the data will be divided into training, validation, and test sets. Any preprocessing that learns statistics from the data should be fitted only on the training portion to avoid data leakage.

Categorical variables will require encoding, numerical variables may require scaling depending on the model, and missing-value strategies will need to be incorporated into a repeatable preprocessing pipeline. Baseline models should be established before more complex methods are introduced. Because `Emergency` work orders are expected to be less common than `Medium` work orders, evaluation should not rely on accuracy alone. Precision, recall, F1 score, per-class performance, and a confusion matrix will be important. Class weighting or other imbalance strategies may also need to be evaluated.

Possible future models include logistic regression, decision trees, random forests, and gradient-boosted trees. Their performance should be compared against a simple baseline and interpreted in the context of maintenance operations rather than only by selecting the largest score.

## Neural Network Preparation Reflection

A neural network would require additional preparation because the maintenance table contains a mixture of categorical, numerical, Boolean, and possibly text features. Categorical values such as asset type, issue type, and maintenance type would need numerical representations such as one-hot encoding or learned embeddings. Numerical inputs such as asset age, repair cost, PM overdue days, and previous failure counts would normally be scaled or normalized so that features with large numerical ranges do not dominate optimization.

The dataset would also need clearly separated training, validation, and test subsets. Missing-value handling must be consistent across those subsets, and any statistics used for imputation or scaling must be learned from training data only. If priority classes are imbalanced, class weights or carefully chosen sampling strategies may be needed. Inputs and targets would ultimately be converted into tensors compatible with the chosen deep-learning framework.

A neural network should not be assumed to be better simply because it is more complex. With a tabular dataset of this size, classical machine-learning models may outperform or equal a neural network while remaining easier to interpret. Model choice should therefore be based on evidence from validation results.

## Agentic Automation Potential Reflection

The cleaned data workflow could eventually support an agentic maintenance system. A future agent could receive a natural-language maintenance request, identify the relevant property and asset, retrieve recent work-order and preventive-maintenance history, call a trained priority model, and present a recommended priority to a facilities manager.

A more advanced workflow could then help create the work order, suggest an appropriate technician or trade, notify responsible staff, monitor status, and request missing information from the requestor. The machine-learning model would be one tool used by the agent rather than the entire decision-making system.

Human oversight would remain important. High-risk decisions involving life safety, emergency response, regulatory requirements, or costly asset replacement should not be silently automated from a synthetic-data-trained model. A production implementation would need real operational validation, access controls, audit logs, confidence thresholds, exception handling, privacy protections, and clear rules for when a human must approve or override an automated recommendation.

## Assumptions and Limitations

The project begins with synthetic data because historical customer maintenance data is not yet available. This creates several important limitations:

- Relationships in the data reflect rules and probabilities chosen by the project, not measured customer behavior.
- Synthetic data can unintentionally encode the assumptions or biases of its designer.
- Model performance on synthetic data cannot demonstrate real-world predictive performance.
- Generated descriptions, costs, failure rates, and priority distributions may differ from those of actual organizations.
- Real maintenance data may contain additional sources of missingness, reporting inconsistency, seasonal behavior, technician effects, property differences, and policy-driven priority decisions that are not represented in the simulation.

Once real operational data becomes available, the synthetic assumptions should be compared with observed distributions and relationships. The preprocessing workflow should then be reevaluated before any production model is trained or deployed.

## Planned Visual Analysis

The notebook will include at least three labeled visualizations with written interpretations. Candidate analyses include:

- work-order distribution by priority;
- repair cost by asset type;
- relationship between asset age and failure history;
- resolution time by priority;
- maintenance condition or PM compliance across asset classes.

Every final chart will include a descriptive title and labeled axes, followed by an interpretation that distinguishes association from causation.

## Project Roadmap

- [ ] Initialize Python repository structure and development environment.
- [ ] Document the synthetic maintenance data specification.
- [ ] Build the reproducible Python data generator.
- [ ] Create the primary Jupyter notebook and ingest data with Pandas.
- [ ] Implement documented data-cleaning functions.
- [ ] Perform exploratory data analysis.
- [ ] Add at least three interpreted visualizations.
- [ ] Complete notebook summary, assumptions, and limitations.
- [ ] Generate `requirements.txt` using `pip freeze`.
- [ ] Verify the notebook runs top-to-bottom without errors.
- [ ] Complete final rubric review.
- [ ] Extend the cleaned dataset into a future priority-classification ML project.

## Git/GitHub Workflow

Development will use multiple meaningful commits and at least one branch beyond `main`, as required by the project rubric. Changes should be grouped into understandable units such as data design, synthetic generation, cleaning, EDA, visualizations, and documentation rather than submitted as one large final commit.

## Academic and Ethical Note

This repository is an educational machine-learning/data-workflow project. Synthetic records are clearly identified as synthetic, and conclusions will be presented with appropriate limitations. No synthetic analysis should be represented as evidence collected from real clients or real maintenance operations.
