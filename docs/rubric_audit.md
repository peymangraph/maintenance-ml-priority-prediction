# AI Programming Foundations Project — Rubric Audit

This file maps each submitted rubric criterion to concrete evidence in the repository.

## Notebook Execution — PASS

- Primary notebook: `notebooks/maintenance_data_workflow.ipynb`
- GitHub Actions workflow: `.github/workflows/notebook-check.yml`
- The clean-environment validation installs `requirements.txt` and executes the notebook from top to bottom with `jupyter nbconvert --execute`.
- Validation run `32188001904` completed successfully on the submission branch.

## Data Ingestion — PASS

Evidence in notebook section **2. Data Ingestion**:

- loads the raw CSV using `pandas.read_csv()`;
- parses `created_date`;
- preserves the valid literal occupancy category `None`;
- reports dataset shape;
- displays the first rows with `df_raw.head()`.

## Data Cleaning Functions — PASS

`src/cleaning.py` contains four reusable functions with docstrings:

1. `standardize_categories()`
2. `remove_duplicate_work_orders()`
3. `flag_invalid_values()`
4. `impute_missing_values()`

All four are imported and used by the notebook.

## Exploratory Analysis Function — PASS

`src/eda.py` contains reusable documented EDA functions:

- `priority_summary()`
- `asset_type_summary()`
- `priority_operational_summary()`
- `high_risk_work_orders()`

The notebook uses grouping, summary statistics, filtering, and descriptive statistics.

## Visualizations — PASS

`src/visualizations.py` and notebook section **5. Visualizations** contain four Matplotlib visualizations:

1. priority distribution;
2. average previous failures by priority;
3. median estimated repair cost by asset type;
4. resolution-time distribution by priority.

Every chart has a descriptive title and labeled axes.

## Reproducibility — PASS

- `requirements.txt` contains pinned package versions derived from `pip freeze`.
- README contains setup and reproduction instructions.
- GitHub Actions successfully installs the requirements in Python 3.11 and executes the notebook in a fresh hosted environment.
- Synthetic generation uses fixed random seed `42`.

## Cleaning Justification — PASS

Notebook section **3. Data Cleaning** explains why category normalization, duplicate removal, invalid-value handling, and imputation are necessary and discusses limitations of those choices.

## Visualization Interpretation — PASS

Each of the four notebook visualizations is immediately followed by a written interpretation. The interpretations distinguish designed synthetic associations from real-world causal claims.

## Summary & Interpretation — PASS

Notebook section **6. Summary and Interpretation** describes:

- main patterns;
- cleaning results;
- candidate future features;
- assumptions;
- synthetic-data limitations;
- generalizability limits;
- future real-data validation requirements;
- target-leakage risk from post-outcome fields such as `resolution_hours`.

## Bias Awareness — PASS

README section **Bias Awareness: How Poor Cleaning Can Introduce Bias** discusses:

- selective row deletion;
- imputation bias;
- inappropriate outlier removal;
- category-normalization risks;
- duplicate-removal risks;
- effects on underrepresented priority and asset groups.

## Required Notebook Sections — PASS

The notebook contains clear Markdown headings for:

1. Setup
2. Data Ingestion
3. Data Cleaning
4. Exploratory Data Analysis
5. Visualizations
6. Summary and Interpretation

## Function Documentation / Readability — PASS

Student-defined functions across generator, cleaning, EDA, and visualization modules contain informative docstrings. Code is separated into reusable modules rather than placed entirely in one notebook.

## README Completeness — PASS

README includes:

- project description;
- synthetic-data disclosure;
- implemented dataset description;
- repository structure;
- reproduction instructions;
- cleaning justification;
- bias reflection;
- ML workflow reflection;
- neural-network preparation reflection;
- agentic automation reflection;
- assumptions and limitations;
- security warning.

## Git/GitHub Usage — PASS

- Default branch: `main`
- Development branch: `feature/project-setup`
- Development was performed through multiple meaningful commits.
- Pull request #13 reviews the submission work before merge.

## Future Integration Reflections — PASS

README contains complete sections for:

- future supervised ML workflow;
- neural-network data preparation;
- agentic maintenance automation potential;
- human oversight and production safeguards.

## Workflow Completeness — PASS

The repository provides a reusable pipeline:

```text
Synthetic generation
    -> raw ingestion
    -> documented cleaning
    -> processed dataset
    -> reusable EDA
    -> interpreted visualizations
    -> future ML foundation
```

The workflow is modular, reproducible, auditable, security-conscious, and explicitly separates synthetic educational results from future real-world validation.
