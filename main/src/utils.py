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

def format_route_result(routes: List[Dict]) -> str:
    """Format the route guidance results as a string."""
    result = ""
    for i, route in enumerate(routes, 1):
        result += f"Route {i}:\n"
        result += f"Path: {' -> '.join(route['path'])}\n"
        result += f"Estimated time: {route['time']:.2f} minutes\n"
        result += f"Distance: {route['distance']:.2f} km\n\n"
    return result
