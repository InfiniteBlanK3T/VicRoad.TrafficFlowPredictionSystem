    import networkx as nx
    import numpy as np


    def calculate_travel_time(volume, distance):
        """Calculate travel time based on volume and distance."""
        base_time = distance / 60  # Assuming 60 km/h speed limit
        congestion_factor = 1 + (volume / 1000)  # Arbitrary congestion factor
        return (
            base_time * congestion_factor + 0.5
        )  # Adding 30 seconds (0.5 minutes) for intersection delay


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
