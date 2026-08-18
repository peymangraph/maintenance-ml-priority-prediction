"""Matplotlib visualizations for the maintenance exploratory-analysis workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .eda import PRIORITY_ORDER, priority_summary


def plot_priority_distribution(df: pd.DataFrame):
    """Plot work-order counts by priority and return the Matplotlib axes."""

    summary = priority_summary(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    summary["count"].plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Maintenance Work Orders by Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Number of Work Orders")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return ax


def plot_failures_by_priority(df: pd.DataFrame):
    """Plot average previous failures for each work-order priority level."""

    failures = (
        df.groupby("priority")["previous_failures_12m"]
        .mean()
        .reindex(PRIORITY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    failures.plot(kind="bar", ax=ax)
    ax.set_title("Average Previous Failures by Work-Order Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Average Failures in Previous 12 Months")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return ax


def plot_median_cost_by_asset_type(df: pd.DataFrame):
    """Plot median estimated repair cost for each canonical asset type."""

    median_cost = (
        df.groupby("asset_type")["estimated_repair_cost"]
        .median()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    median_cost.plot(kind="bar", ax=ax)
    ax.set_title("Median Estimated Repair Cost by Asset Type")
    ax.set_xlabel("Asset Type")
    ax.set_ylabel("Median Estimated Repair Cost ($)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return ax


def plot_resolution_by_priority(df: pd.DataFrame):
    """Plot resolution-time distributions by priority without extreme fliers."""

    groups = [
        df.loc[df["priority"] == priority, "resolution_hours"]
        for priority in PRIORITY_ORDER
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=PRIORITY_ORDER, showfliers=False)
    ax.set_title("Resolution Time Distribution by Work-Order Priority")
    ax.set_xlabel("Priority")
    ax.set_ylabel("Resolution Hours")
    fig.tight_layout()
    return ax
