# Raw Synthetic Data

The raw maintenance CSV used by this project is generated reproducibly by `src/generate_data.py` with the fixed random seed defined in `src/config.py`.

Run the generator from the repository root:

```bash
python -m src.generate_data
```

This creates:

```text
data/raw/maintenance_work_orders.csv
```

The default generator creates 8,000 base synthetic work orders and intentionally adds a small set of duplicate records and other documented data-quality issues for the cleaning exercise.

The data is entirely synthetic. Do not place real customer maintenance records, personally identifiable information, credentials, API keys, passwords, tokens, or production connection strings in this directory or repository.
