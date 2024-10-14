import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List


def format_prediction_result(
    street: str, date: str, time: str, prediction: float, actual: float
) -> str:
    """Format the prediction result as a string."""
    return (
        f"Street: {street}\n"
        f"Date: {date}\n"
        f"Time: {time}\n"
        f"{'Predicted:':>10} {prediction:.2f} vehicles\n"
        f"{'Actual:':>10} {actual:.2f} vehicles\n"
        f"{'Difference:':>10} {abs(prediction - actual):.2f} vehicles"
    )


def plot_prediction(
    ax: plt.Axes,
    street_data: pd.DataFrame,
    prediction: float,
    actual: float,
    selected_time: str,
):
    """Plot the prediction results."""
    # Handle both timedelta and datetime objects in the 'Time' column
    if isinstance(street_data["Time"].iloc[0], pd.Timedelta):
        times = street_data["Time"].apply(lambda x: str(x).split()[-1])
    else:
        times = street_data["Time"].dt.strftime("%H:%M")

    actual_data = street_data["TrafficVolume"].values

    ax.plot(times, actual_data, label="Actual", color="blue", alpha=0.7)
    ax.scatter(
        selected_time, prediction, color="red", s=100, zorder=5, label="Prediction"
    )
    ax.scatter(
        selected_time, actual, color="green", s=100, zorder=5, label="Actual (selected)"
    )

    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Traffic Volume (vehicles per 15 min)")
    ax.set_title(f'24-Hour Traffic Volume Prediction - {street_data["Street"].iloc[0]}')
    ax.legend()

    # Show every 3 hours
    ax.set_xticks(times[::12])
    ax.set_xticklabels(times[::12], rotation=45, ha="right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)


def highlight_time_range(ax: plt.Axes, start: str, end: str, label: str):
    """Highlight a time range on the plot."""
    ax.axvspan(start, end, alpha=0.2, color="yellow")
    ax.text(
        start,
        ax.get_ylim()[1],
        label,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )


def format_route_result(routes: List[Dict]) -> str:
    """Format the route guidance results as a string."""
    result = ""
    for i, route in enumerate(routes, 1):
        result += f"Route {i}:\n"
        result += f"Path: {' -> '.join(route['path'])}\n"
        result += f"Estimated time: {route['time']:.2f} minutes\n"
        result += f"Distance: {route['distance']:.2f} km\n\n"
    return result
