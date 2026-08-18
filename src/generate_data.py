"""Generate a reproducible synthetic CMMS-style maintenance work-order dataset.

The dataset is intentionally synthetic. It includes plausible maintenance-domain
relationships plus controlled missing values, formatting inconsistencies,
duplicates, and outliers so the course cleaning workflow has meaningful work to
do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import RANDOM_SEED, RAW_DATA_FILE
except ImportError:  # Allows `python src/generate_data.py` as well as module execution.
    from config import RANDOM_SEED, RAW_DATA_FILE


ASSET_TYPES = np.array(
    [
        "HVAC",
        "Electrical",
        "Plumbing",
        "Elevator",
        "Fire/Life Safety",
        "Lighting",
        "Pump",
        "Appliance",
        "Access Control",
        "Building Envelope",
    ],
    dtype=object,
)

ASSET_TYPE_PROBS = np.array(
    [0.20, 0.15, 0.15, 0.05, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05]
)

ISSUE_TYPES = {
    "HVAC": ["No Cooling", "No Heating", "Unusual Noise", "Airflow Problem", "Thermostat Issue"],
    "Electrical": ["Power Loss", "Breaker Trip", "Outlet Failure", "Lighting Circuit", "Overheating"],
    "Plumbing": ["Leak", "Clog", "Low Pressure", "Fixture Failure", "Water Temperature"],
    "Elevator": ["Out of Service", "Door Fault", "Leveling Issue", "Unusual Noise", "Control Fault"],
    "Fire/Life Safety": ["Alarm Fault", "Detector Fault", "Panel Trouble", "Inspection Finding", "Device Damage"],
    "Lighting": ["Lamp Failure", "Fixture Failure", "Control Issue", "Emergency Lighting", "Flickering"],
    "Pump": ["No Flow", "Leak", "Motor Fault", "Low Pressure", "Unusual Noise"],
    "Appliance": ["No Power", "Poor Performance", "Leak", "Unusual Noise", "Control Fault"],
    "Access Control": ["Door Not Locking", "Reader Failure", "Door Not Opening", "Alarm Condition", "Hardware Damage"],
    "Building Envelope": ["Roof Leak", "Window Leak", "Door Draft", "Exterior Damage", "Water Intrusion"],
}

BASE_COST = {
    "HVAC": 900,
    "Electrical": 600,
    "Plumbing": 500,
    "Elevator": 1800,
    "Fire/Life Safety": 750,
    "Lighting": 250,
    "Pump": 850,
    "Appliance": 350,
    "Access Control": 450,
    "Building Envelope": 1100,
}

CRITICALITY_PROBS = {
    "HVAC": [0.45, 0.40, 0.15],
    "Electrical": [0.40, 0.40, 0.20],
    "Plumbing": [0.55, 0.35, 0.10],
    "Elevator": [0.10, 0.35, 0.55],
    "Fire/Life Safety": [0.05, 0.20, 0.75],
    "Lighting": [0.75, 0.20, 0.05],
    "Pump": [0.35, 0.40, 0.25],
    "Appliance": [0.80, 0.18, 0.02],
    "Access Control": [0.45, 0.40, 0.15],
    "Building Envelope": [0.65, 0.30, 0.05],
}

SAFETY_BASE = {
    "HVAC": 0.04,
    "Electrical": 0.13,
    "Plumbing": 0.05,
    "Elevator": 0.18,
    "Fire/Life Safety": 0.28,
    "Lighting": 0.03,
    "Pump": 0.06,
    "Appliance": 0.03,
    "Access Control": 0.08,
    "Building Envelope": 0.04,
}


def _condition_for_age(age_years: float, rng: np.random.Generator) -> str:
    """Sample asset condition with probabilities that worsen as age increases."""
    if age_years < 5:
        probs = [0.78, 0.19, 0.03]
    elif age_years < 12:
        probs = [0.55, 0.37, 0.08]
    elif age_years < 20:
        probs = [0.30, 0.50, 0.20]
    else:
        probs = [0.15, 0.45, 0.40]

    return str(rng.choice(["Good", "Fair", "Poor"], p=probs))


def _generate_assets(
    rng: np.random.Generator,
    num_properties: int,
    num_assets: int,
) -> pd.DataFrame:
    """Create the simulated asset inventory used by the work-order generator."""
    asset_types = rng.choice(ASSET_TYPES, size=num_assets, p=ASSET_TYPE_PROBS)
    ages = np.clip(rng.gamma(shape=2.2, scale=5.2, size=num_assets), 0.2, 35.0)
    properties = rng.integers(1, num_properties + 1, size=num_assets)

    conditions = [_condition_for_age(age, rng) for age in ages]
    criticalities = [
        rng.choice(["Standard", "Important", "Critical"], p=CRITICALITY_PROBS[asset_type])
        for asset_type in asset_types
    ]

    assets = pd.DataFrame(
        {
            "asset_id": [f"AST-{i:04d}" for i in range(1, num_assets + 1)],
            "property_id": [f"PROP-{i:02d}" for i in properties],
            "asset_type": asset_types,
            "base_asset_age_years": ages.round(1),
            "asset_condition": conditions,
            "asset_criticality": criticalities,
        }
    )

    condition_factor = assets["asset_condition"].map(
        {"Good": 0.7, "Fair": 1.0, "Poor": 1.6}
    ).to_numpy()
    age_factor = 0.6 + assets["base_asset_age_years"].to_numpy() / 20.0
    assets["failure_weight"] = np.clip(age_factor * condition_factor, 0.2, None)
    return assets


def _inject_quality_issues(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject limited missingness, formatting inconsistencies, duplicates, and outliers."""
    dirty = df.copy()

    for column, rate in {
        "asset_condition": 0.025,
        "days_since_last_service": 0.03,
        "estimated_repair_cost": 0.02,
    }.items():
        idx = rng.choice(
            dirty.index,
            size=max(1, int(len(dirty) * rate)),
            replace=False,
        )
        dirty.loc[idx, column] = np.nan

    asset_idx = rng.choice(
        dirty.index,
        size=max(1, int(len(dirty) * 0.02)),
        replace=False,
    )
    for idx in asset_idx:
        value = str(dirty.at[idx, "asset_type"])
        dirty.at[idx, "asset_type"] = rng.choice(
            [value.lower(), value.upper(), f" {value} "]
        )

    priority_idx = rng.choice(
        dirty.index,
        size=max(1, int(len(dirty) * 0.015)),
        replace=False,
    )
    for idx in priority_idx:
        value = str(dirty.at[idx, "priority"])
        dirty.at[idx, "priority"] = rng.choice(
            [value.lower(), value.upper(), f" {value} "]
        )

    age_idx = rng.choice(
        dirty.index,
        size=max(1, int(len(dirty) * 0.003)),
        replace=False,
    )
    dirty.loc[age_idx, "asset_age_years"] = -rng.uniform(
        0.5, 3.0, size=len(age_idx)
    ).round(1)

    resolution_idx = rng.choice(
        dirty.index,
        size=max(1, int(len(dirty) * 0.003)),
        replace=False,
    )
    dirty.loc[resolution_idx, "resolution_hours"] = rng.uniform(
        500, 1200, size=len(resolution_idx)
    ).round(1)

    cost_idx = rng.choice(
        dirty.index,
        size=max(1, int(len(dirty) * 0.002)),
        replace=False,
    )
    dirty.loc[cost_idx, "estimated_repair_cost"] = rng.uniform(
        40000, 100000, size=len(cost_idx)
    ).round(2)

    duplicate_rows = dirty.sample(frac=0.005, random_state=RANDOM_SEED)
    dirty = pd.concat([dirty, duplicate_rows], ignore_index=True)

    return dirty.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def generate_dataset(
    num_properties: int = 10,
    num_assets: int = 750,
    num_work_orders: int = 8000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a reproducible synthetic maintenance work-order dataset.

    Parameters
    ----------
    num_properties:
        Number of simulated properties represented in the portfolio.
    num_assets:
        Number of simulated maintainable assets.
    num_work_orders:
        Number of base work orders generated before duplicate injection.
    random_seed:
        Seed controlling all pseudo-random generation.

    Returns
    -------
    pandas.DataFrame
        Raw synthetic work-order data containing intentional quality issues.
    """
    rng = np.random.default_rng(random_seed)
    assets = _generate_assets(rng, num_properties, num_assets)

    failure_probs = assets["failure_weight"].to_numpy()
    failure_probs = failure_probs / failure_probs.sum()
    selected_idx = rng.choice(
        assets.index.to_numpy(),
        size=num_work_orders,
        replace=True,
        p=failure_probs,
    )
    selected = assets.loc[selected_idx].reset_index(drop=True)

    start = np.datetime64("2023-01-01")
    end = np.datetime64("2026-01-01")
    max_days = int((end - start) / np.timedelta64(1, "D"))
    date_offsets = rng.integers(0, max_days, size=num_work_orders)
    created_dates = start + date_offsets.astype("timedelta64[D]")

    maintenance_type = rng.choice(
        ["Corrective", "Preventive", "Inspection"],
        size=num_work_orders,
        p=[0.66, 0.24, 0.10],
    )
    issue_type = [rng.choice(ISSUE_TYPES[asset_type]) for asset_type in selected["asset_type"]]

    days_to_end = (end - created_dates).astype("timedelta64[D]").astype(int)
    event_age = np.maximum(
        0.1,
        selected["base_asset_age_years"].to_numpy() - days_to_end / 365.25,
    )

    condition_lambda = selected["asset_condition"].map(
        {"Good": 0.4, "Fair": 1.2, "Poor": 2.6}
    ).to_numpy()
    age_lambda = np.clip(event_age / 20.0, 0, 1.8)
    failures = rng.poisson(np.clip(condition_lambda + age_lambda, 0.05, 5.0))

    service_days = rng.gamma(shape=2.0, scale=55.0, size=num_work_orders)
    service_days += failures * rng.uniform(4, 18, size=num_work_orders)
    service_days = np.clip(service_days, 1, 500)

    service_interval = np.where(
        selected["asset_criticality"].eq("Critical").to_numpy(),
        90,
        np.where(
            selected["asset_criticality"].eq("Important").to_numpy(),
            150,
            210,
        ),
    )
    overdue = np.maximum(
        0,
        service_days - service_interval + rng.normal(0, 15, size=num_work_orders),
    )

    occupancy_score = (
        (maintenance_type == "Corrective").astype(float) * 0.7
        + selected["asset_criticality"].map(
            {"Standard": 0.0, "Important": 0.4, "Critical": 0.9}
        ).to_numpy()
        + failures * 0.10
        + rng.normal(0, 0.8, size=num_work_orders)
    )
    occupancy_impact = np.select(
        [
            occupancy_score < -0.1,
            occupancy_score < 0.65,
            occupancy_score < 1.35,
        ],
        ["None", "Low", "Medium"],
        default="High",
    )

    safety_prob = np.array(
        [SAFETY_BASE[asset_type] for asset_type in selected["asset_type"]],
        dtype=float,
    )
    safety_prob += selected["asset_criticality"].map(
        {"Standard": 0.0, "Important": 0.02, "Critical": 0.07}
    ).to_numpy()
    safety_prob += np.where(maintenance_type == "Corrective", 0.02, 0.0)
    safety_prob = np.clip(safety_prob, 0.01, 0.55)
    safety_related = rng.random(num_work_orders) < safety_prob

    base_cost = np.array(
        [BASE_COST[asset_type] for asset_type in selected["asset_type"]],
        dtype=float,
    )
    condition_cost = selected["asset_condition"].map(
        {"Good": 0.85, "Fair": 1.0, "Poor": 1.35}
    ).to_numpy()
    maintenance_cost = np.where(
        maintenance_type == "Corrective",
        1.15,
        np.where(maintenance_type == "Preventive", 0.65, 0.45),
    )
    cost_noise = rng.lognormal(mean=0.0, sigma=0.45, size=num_work_orders)
    estimated_cost = (
        base_cost
        * condition_cost
        * maintenance_cost
        * (1 + failures * 0.08)
        * cost_noise
    )
    estimated_cost = np.clip(estimated_cost, 40, 25000)

    urgency = (
        rng.normal(0, 0.85, size=num_work_orders)
        + safety_related.astype(float) * 2.3
        + pd.Series(occupancy_impact).map(
            {"None": -0.3, "Low": 0.0, "Medium": 0.55, "High": 1.25}
        ).to_numpy()
        + selected["asset_criticality"].map(
            {"Standard": 0.0, "Important": 0.35, "Critical": 0.85}
        ).to_numpy()
        + selected["asset_condition"].map(
            {"Good": 0.0, "Fair": 0.25, "Poor": 0.75}
        ).to_numpy()
        + np.minimum(failures, 5) * 0.16
        + np.minimum(overdue, 240) / 240.0 * 0.55
        + np.where(
            maintenance_type == "Corrective",
            0.35,
            np.where(maintenance_type == "Inspection", -0.15, -0.3),
        )
    )

    q22, q69, q94 = np.quantile(urgency, [0.22, 0.69, 0.94])
    priority = np.select(
        [urgency <= q22, urgency <= q69, urgency <= q94],
        ["Low", "Medium", "High"],
        default="Emergency",
    ).astype(object)

    perturb_idx = rng.choice(
        np.arange(num_work_orders),
        size=int(num_work_orders * 0.04),
        replace=False,
    )
    levels = np.array(["Low", "Medium", "High", "Emergency"], dtype=object)
    for idx in perturb_idx:
        current = priority[idx]
        priority[idx] = rng.choice(levels[levels != current])

    complexity = (
        1.0
        + selected["asset_condition"].map(
            {"Good": 0.0, "Fair": 0.4, "Poor": 0.9}
        ).to_numpy()
        + failures * 0.12
        + np.log1p(estimated_cost) / 9.0
    )
    priority_response = pd.Series(priority).map(
        {"Low": 1.15, "Medium": 1.0, "High": 0.85, "Emergency": 0.70}
    ).to_numpy()
    resolution_hours = (
        rng.gamma(shape=2.2, scale=5.0, size=num_work_orders)
        * complexity
        * priority_response
    )
    resolution_hours = np.clip(resolution_hours, 0.5, 240.0)

    df = pd.DataFrame(
        {
            "work_order_id": [f"WO-{i:06d}" for i in range(1, num_work_orders + 1)],
            "property_id": selected["property_id"].to_numpy(),
            "asset_id": selected["asset_id"].to_numpy(),
            "created_date": pd.to_datetime(created_dates),
            "asset_type": selected["asset_type"].to_numpy(),
            "asset_age_years": np.round(event_age, 1),
            "asset_condition": selected["asset_condition"].to_numpy(),
            "asset_criticality": selected["asset_criticality"].to_numpy(),
            "maintenance_type": maintenance_type,
            "issue_type": issue_type,
            "previous_failures_12m": failures,
            "days_since_last_service": np.round(service_days, 0),
            "pm_overdue_days": np.round(overdue, 0),
            "occupancy_impact": occupancy_impact,
            "safety_related": safety_related,
            "estimated_repair_cost": np.round(estimated_cost, 2),
            "resolution_hours": np.round(resolution_hours, 1),
            "priority": priority,
        }
    )

    return _inject_quality_issues(df, rng)


def save_dataset(
    output_path: str | Path = RAW_DATA_FILE,
    **generation_kwargs: int,
) -> pd.DataFrame:
    """Generate the raw dataset, save it as CSV, and return the DataFrame."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(**generation_kwargs)
    df.to_csv(output, index=False)
    return df


def main() -> None:
    """Generate the default dataset and print a short reproducibility summary."""
    df = save_dataset()
    normalized_priority = df["priority"].astype(str).str.strip().str.title()

    print(f"Saved {len(df):,} raw rows to {RAW_DATA_FILE}")
    print("Priority distribution:")
    print((normalized_priority.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print("Missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
