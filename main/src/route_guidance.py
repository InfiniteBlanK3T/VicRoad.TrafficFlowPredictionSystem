import networkx as nx
import numpy as np
import math


def calculate_travel_time(volume, distance):
    """
    Calculate travel time based on volume and distance.
    Assumptions:
    1. 64 km/h free flow speed
    2. 1500 max capacity for each SCATS site
    """

    free_flow_speed = 64.0
    capacity_speed = free_flow_speed / 2
    max_flow = 1500.0

    if volume > max_flow:
        raise ValueError(f"Flow rate exceeds max flow assumption")

    var_a = -1.0 * max_flow / (capacity_speed**2)
    var_b = -2.0 * capacity_speed * var_a

    # Given equation: x = var_a * y**2 + var_b * y, plug into quadratic formula
    # Calculate the discriminant
    discriminant = var_b**2 - 4 * var_a * volume * -1

    traffic_speeds = []

    # At max flow
    if discriminant == 0:
        traffic_speed1 = -1 * var_b / (2 * var_a)
        traffic_speeds.append(traffic_speed1)
    else:
        # if under congested
        traffic_speed1 = (-var_b - math.sqrt(discriminant)) / (2 * var_a)
        traffic_speeds.append(traffic_speed1)
        # if over congested
        traffic_speed2 = (-var_b + math.sqrt(discriminant)) / (2 * var_a)
        traffic_speeds.append(traffic_speed2)

    # Need to add way to determine if traffic is over or under maximum flow, using default of under
    traffic_time = 60 * (distance / traffic_speeds[0]) + 0.5
    # Adding 30 seconds (0.5 minutes) for intersection delay

    return (
        traffic_time
    )


def create_graph(df, street_segments):
    G = nx.Graph()

    # Add all SCATS numbers as nodes
    for scats, data in df.groupby("SCATS Number"):
        G.add_node(
            str(scats), pos=(data["NB_LATITUDE"].iloc[0], data["NB_LONGITUDE"].iloc[0])
        )

    # Connect nodes based on street segments
    for street, segment in street_segments.items():
        coords = segment["coords"]
        avg_traffic = segment["avg_traffic"]
        scats_on_street = df[df["Street"] == street]["SCATS Number"].unique()

        for i in range(len(scats_on_street) - 1):
            start = str(scats_on_street[i])
            end = str(scats_on_street[i + 1])
            if start in G and end in G:
                start_pos = G.nodes[start]["pos"]
                end_pos = G.nodes[end]["pos"]
                distance = np.sqrt(
                    (start_pos[0] - end_pos[0]) ** 2 + (start_pos[1] - end_pos[1]) ** 2
                )
                travel_time = calculate_travel_time(avg_traffic, distance)
                G.add_edge(start, end, weight=travel_time, distance=distance)

    return G


def find_routes(G, origin, destination, k=5):
    origin = str(origin)
    destination = str(destination)

    if origin not in G:
        raise ValueError(f"Origin SCATS number {origin} not found in the graph.")
    if destination not in G:
        raise ValueError(
            f"Destination SCATS number {destination} not found in the graph."
        )

    try:
        routes = list(
            nx.shortest_simple_paths(G, origin, destination, weight="weight")
        )[:k]
    except nx.NetworkXNoPath:
        raise ValueError(f"No path found between {origin} and {destination}.")

    formatted_routes = []
    for route in routes:
        total_time = sum(
            G[route[i]][route[i + 1]]["weight"] for i in range(len(route) - 1)
        )
        total_distance = sum(
            G[route[i]][route[i + 1]]["distance"] for i in range(len(route) - 1)
        )
        formatted_routes.append(
            {"path": route, "time": total_time, "distance": total_distance}
        )

    return formatted_routes


def route_guidance(df, street_segments, origin, destination):
    G = create_graph(df, street_segments)
    routes = find_routes(G, origin, destination)
    return routes
