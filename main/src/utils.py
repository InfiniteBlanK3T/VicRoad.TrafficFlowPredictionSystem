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

def format_route_result(routes, df):
    """
    Format the route guidance results as a string with street names.
    
    Args:
        routes (list): List of route dictionaries
        df (pd.DataFrame): DataFrame containing SCATS data
        
    Returns:
        str: Formatted route information
    """
    result = ""
    for i, route in enumerate(routes, 1):
        # Get the street names for each SCATS number in the path
        path_details = []
        for scats in route['path']:
            scats_data = df[df['SCATS Number'] == int(scats)].iloc[0]
            intersection = scats_data['Location'].replace('_', ' ').replace('/', ' & ')
            path_details.append(intersection)
            
        result += f"Route {i}:\n"
        result += "Path:\n"
        for j, intersection in enumerate(path_details):
            if j == 0:
                result += f"  Start: {intersection}\n"
            elif j == len(path_details) - 1:
                result += f"  End: {intersection}\n"
            else:
                result += f"  Via: {intersection}\n"
                
        result += f"Estimated time: {route['time']:.1f} minutes\n"
        result += f"Distance: {route['distance']:.2f} km\n\n"
    
    return result
