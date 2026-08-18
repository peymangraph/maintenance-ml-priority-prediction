"""Reusable exploratory-analysis helpers for maintenance work-order data.

These functions are intentionally small and transparent so the notebook can
show how Pandas grouping, summarization, and filtering support a reusable data
workflow before model training begins.
"""

from __future__ import annotations

import pandas as pd


PRIORITY_ORDER = ["Low", "Medium", "High", "Emergency"]


def priority_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return work-order counts and percentages for each priority level.

    Parameters
    ----------
    df:
        Cleaned maintenance work-order DataFrame containing a ``priority``
        column.

    Returns
    -------
    pandas.DataFrame
        One row per priority with ``count`` and ``percentage`` columns ordered
        from Low through Emergency.
    """

    counts = df["priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    percentages = (
        df["priority"].value_counts(normalize=True)
        .reindex(PRIORITY_ORDER, fill_value=0)
        .mul(100)
        .round(2)
    )

    return pd.DataFrame({"count": counts, "percentage": percentages})


def asset_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize work-order volume, failures, cost, and resolution by asset type.

    The function groups maintenance records by asset type and calculates metrics
    useful for identifying operationally important asset categories without
    implying that descriptive associations are causal.
    """

    summary = (
        df.groupby("asset_type")
        .agg(
            work_orders=("work_order_id", "count"),
            avg_asset_age_years=("asset_age_years", "mean"),
            avg_previous_failures=("previous_failures_12m", "mean"),
            median_repair_cost=("estimated_repair_cost", "median"),
            median_resolution_hours=("resolution_hours", "median"),
        )
        .round(2)
        .sort_values("work_orders", ascending=False)
    )

    return summary


def priority_operational_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compare maintenance severity indicators across priority groups.

    Returns average failure history, PM overdue days, repair cost, resolution
    time, and the percentage of safety-related work orders for each priority.
    """

    working = df.copy()
    working["safety_related"] = working["safety_related"].astype(bool)

    summary = (
        working.groupby("priority")
        .agg(
            work_orders=("work_order_id", "count"),
            avg_previous_failures=("previous_failures_12m", "mean"),
            avg_pm_overdue_days=("pm_overdue_days", "mean"),
            median_repair_cost=("estimated_repair_cost", "median"),
            median_resolution_hours=("resolution_hours", "median"),
            safety_related_rate=("safety_related", "mean"),
        )
        .reindex(PRIORITY_ORDER)
    )

    summary["safety_related_pct"] = summary.pop("safety_related_rate") * 100
    return summary.round(2)


def high_risk_work_orders(
    df: pd.DataFrame,
    minimum_failures: int = 3,
    minimum_overdue_days: float = 30,
) -> pd.DataFrame:
    """Filter records with repeated failures and materially overdue maintenance.

    This exploratory filter is not a production risk score. It simply identifies
    records that satisfy two transparent conditions so their priority, asset
    type, and other characteristics can be inspected during EDA.
    """

    mask = (
        (df["previous_failures_12m"] >= minimum_failures)
        & (df["pm_overdue_days"] >= minimum_overdue_days)
    )

    columns = [
        "work_order_id",
        "asset_type",
        "asset_age_years",
        "asset_condition",
        "previous_failures_12m",
        "pm_overdue_days",
        "safety_related",
        "priority",
    ]
    return df.loc[mask, columns].sort_values(
        ["previous_failures_12m", "pm_overdue_days"],
        ascending=False,
    )
