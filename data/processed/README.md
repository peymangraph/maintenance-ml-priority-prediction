# Processed Data

The main notebook writes the cleaned maintenance dataset to:

```text
data/processed/maintenance_ml.csv
```

The processed file is derived from the synthetic raw dataset by applying the documented cleaning functions in `src/cleaning.py`.

The notebook preserves the raw data unchanged and records the effects of cleaning before producing this processed output. This separation supports reproducibility and makes it easier to audit how cleaning decisions affect the dataset.
