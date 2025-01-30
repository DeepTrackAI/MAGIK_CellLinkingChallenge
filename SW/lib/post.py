import os
import glob
from PIL import Image
from scipy.ndimage import distance_transform_edt

import numpy as np
import networkx as nx

import subprocess


def prune_graph(graph):
    """
    Prune the graph by removing edges with low probability. If the ratio between the two highest probabilities
    is greater than 0.8, keep both edges. Otherwise, keep only the edge with the highest probability.

    Args:
    graph (networkx.DiGraph): Predicted graph.

    Returns:
    networkx.DiGraph: Pruned graph.
    """

    pruned_graph = nx.DiGraph()
    pruned_graph.add_nodes_from(
        graph.nodes(data=True)
    )  # Add nodes with their attributes

    # Iterate over all nodes in the graph
    for node in graph.nodes:
        # Get all successors of the node
        successors = list(graph.successors(node))

        # Filter successors to only those in the future
        future_nodes = [
            (s, graph.nodes[s]["frame"])
            for s in successors
            if graph.nodes[s]["frame"] > graph.nodes[node]["frame"]
        ]

        # If there are no future nodes, continue
        if not future_nodes:
            continue

        # Find the minimum frame difference
        min_frame = min(future_nodes, key=lambda x: x[1])[1]

        # Get all nodes in the closest future frame
        nodes_in_min_frame = [s for s, f in future_nodes if f == min_frame]

        if len(nodes_in_min_frame) <= 1:
            # If there's only one edge, add it directly
            # successor, probability, edge_data = edge_list_sorted[0]
            # pruned_graph.add_edge(node, successor, **edge_data)
            for successor in nodes_in_min_frame:
                edge_data = graph.get_edge_data(node, successor)
                pruned_graph.add_edge(node, successor, **edge_data)

        elif len(nodes_in_min_frame) > 1:
            # List to hold the (successor, probability) pairs
            edge_list = []
            # Add edges with attributes (like 'probability') to a list for sorting
            for successor in nodes_in_min_frame:
                edge_data = graph.get_edge_data(node, successor)
                probability = edge_data.get("probability", 0)
                edge_list.append((successor, probability, edge_data))

            # Sort edges by probability in descending order
            edge_list_sorted = sorted(edge_list, key=lambda x: x[1], reverse=True)

            # Get the two highest probabilities
            top_prob = edge_list_sorted[0][1]
            second_prob = edge_list_sorted[1][1]

            # Calculate the ratio between the top two probabilities
            prob_ratio = second_prob / top_prob

            if prob_ratio > 0.8:
                # Keep both edges if the ratio is greater than 0.8
                for successor, probability, edge_data in edge_list_sorted[:2]:
                    pruned_graph.add_edge(node, successor, **edge_data)
            else:
                # Otherwise, keep only the edge with the highest probability
                successor, probability, edge_data = edge_list_sorted[0]
                pruned_graph.add_edge(node, successor, **edge_data)

    return pruned_graph


def process_graph_based_on_probability(G):
    """
    Process the graph by eliminating one edge if a receiver node has multiple incoming edges.
    The edge with the lower 'probability' value is eliminated.

    Args:
    G (networkx.DiGraph): Directed graph with a 'probability' feature on edges.

    Returns:
    networkx.DiGraph: Processed graph with edges removed based on 'probability'.
    """
    # Create a copy of the graph to modify it
    G_processed = G.copy()

    # Iterate over each node to find those that receive edges
    for receiver in G.nodes:
        # Get the incoming edges (from sender nodes to this receiver)
        incoming_edges = list(G.in_edges(receiver, data=True))

        # If the receiver has more than one incoming edge, process it
        if len(incoming_edges) > 1:
            # Sort the incoming edges by the 'probability' attribute in descending order
            incoming_edges_sorted = sorted(
                incoming_edges, key=lambda x: x[2]["probability"], reverse=True
            )

            # Keep the edge with the highest probability, remove the others
            for sender, receiver, data in incoming_edges_sorted[1:]:
                G_processed.remove_edge(sender, receiver)

    return G_processed


def extract_trajectories(pruned_G, source_dir, target_dir, temp_dir, digits):
    """
    Extract trajectories from the pruned graph.

    Args:
    pruned_G (networkx.DiGraph): The pruned directed graph.
    source_dir (str): Directory where the source images are stored.
    target_dir (str): Directory where the processed segmented images are saved.
    digits (int): Number of digits to be used in the filename for padding.

    Returns:
    list: List of trajectories.
    """
    # Initialize parent index vector with zeros for all nodes
    parent_trajectory = np.zeros(len(pruned_G.nodes), dtype=int)

    # Extract trajectories from the pruned graph
    trajectories = []
    trajectory_index = 0  # Start trajectory index from 1
    node_to_trajectory = {}  # Dictionary to track the trajectory of each node

    # Iterate over nodes to initialize root trajectories
    for node in sorted(pruned_G.nodes, key=lambda n: pruned_G.nodes[n]["frame"]):
        if node in node_to_trajectory:
            # Skip nodes that are already part of a trajectory
            continue

        # Start a new trajectory
        current_node = node
        trajectory_index += 1
        current_trajectory_index = trajectory_index
        starting_frame = pruned_G.nodes[current_node]["frame"]

        # Set parent index for the new trajectory from the parent_trajectory vector
        current_parent = parent_trajectory[current_node]

        # Initialize trajectory nodes list
        trajectory_nodes = [current_node]

        # Process the segmented image for the starting node
        node_position = pruned_G.nodes[current_node]["position"]
        current_frame = pruned_G.nodes[current_node]["frame"]
        process_segmented_image(
            starting_frame,
            node_position,
            current_trajectory_index,
            source_dir,
            target_dir,
            temp_dir,
            digits,
        )

        # print(starting_frame, trajectory_index, node_position)

        # Follow the trajectory until there are no more successors or until a split or discontinuity occurs
        while True:
            successors = list(pruned_G.successors(current_node))

            if len(successors) == 0:
                # No further connections, end of trajectory
                last_frame = pruned_G.nodes[current_node]["frame"]
                trajectories.append(
                    {
                        "index": current_trajectory_index,
                        "starting_frame": starting_frame,
                        "last_frame": last_frame,
                        "parent_trajectory": current_parent,
                    }
                )
                # Mark nodes as visited in the trajectory
                for n in trajectory_nodes:
                    node_to_trajectory[n] = current_trajectory_index
                break

            elif len(successors) == 1:
                successor = successors[0]
                successor_frame = pruned_G.nodes[successor]["frame"]
                current_frame = pruned_G.nodes[current_node]["frame"]

                # If frame difference is > 1, start a new trajectory (treat as a daughter)
                if successor_frame - current_frame > 1:
                    # similar to a cell division
                    last_frame = pruned_G.nodes[current_node]["frame"]
                    trajectories.append(
                        {
                            "index": current_trajectory_index,
                            "starting_frame": starting_frame,
                            "last_frame": last_frame,
                            "parent_trajectory": current_parent,
                        }
                    )
                    # Mark nodes as visited in the trajectory
                    for n in trajectory_nodes:
                        node_to_trajectory[n] = current_trajectory_index

                    # Update parent index for each successor node
                    for successor in successors:
                        parent_trajectory[successor] = current_trajectory_index
                    break

                else:
                    # Continue the trajectory with the successor
                    current_node = successor
                    trajectory_nodes.append(current_node)

                    # Process the segmented image for the current node
                    node_position = pruned_G.nodes[current_node]["position"]
                    process_segmented_image(
                        successor_frame,
                        node_position,
                        current_trajectory_index,
                        source_dir,
                        target_dir,
                        temp_dir,
                        digits,
                    )

            elif len(successors) == 2:
                # Cell division occurs; end the current trajectory and start two new trajectories
                last_frame = pruned_G.nodes[current_node]["frame"]
                trajectories.append(
                    {
                        "index": current_trajectory_index,
                        "starting_frame": starting_frame,
                        "last_frame": last_frame,
                        "parent_trajectory": current_parent,
                    }
                )
                # Mark nodes as visited in the trajectory
                for n in trajectory_nodes:
                    node_to_trajectory[n] = current_trajectory_index

                # Update parent index for each successor node
                for successor in successors:
                    parent_trajectory[successor] = current_trajectory_index
                break
            elif len(successors) > 2:
                print(
                    "Warning: There are more than two successors for a node. This is unexpected."
                )

    # Writing output to txt file with all values as integers
    out_file_name = "res_track.txt"
    with open(os.path.join(target_dir, out_file_name), "w") as f:
        for traj in trajectories:
            f.write(
                f"{traj['index']} {int(traj['starting_frame'])} {int(traj['last_frame'])} {traj['parent_trajectory']}\n"
            )


def process_segmented_image(
    frame, position, trajectory_index, source_dir, target_dir, temp_dir, digits
):
    """
    Process the segmented image for the given frame.

    Args:
    frame (int): Frame number of the current node.
    position (tuple): Position (x, y) of the current node.
    trajectory_index (int): Index of the current trajectory.
    target_dir (str): Directory where the processed images are saved.

    Returns:
    None
    """
    # Load source image for current frame
    temp_image_path = os.path.join(
        temp_dir, f"temp_mask{str(int(frame)).zfill(digits)}.tif"
    )
    fallback_image_path = os.path.join(
        source_dir,
        f"mask{str(int(frame)).zfill(digits)}.tif",
    )

    # Check if the temp image exists, otherwise use the fallback
    image_path_to_load = (
        temp_image_path if os.path.exists(temp_image_path) else fallback_image_path
    )

    # Open the image and convert to numpy array
    source_image = Image.open(image_path_to_load)
    source_array = np.array(source_image)

    # Use node position to find pixel value of the image at that position
    y, x = np.round(position).astype(int)
    pixel_value = (
        source_array[y, x]
        if (0 <= x < source_array.shape[1]) and (0 <= y < source_array.shape[0])
        else 0
    )

    # Load segmented image for current frame
    target_image_path = os.path.join(
        target_dir, f"mask{str(int(frame)).zfill(digits)}.tif"
    )

    if os.path.exists(target_image_path):
        target_image = Image.open(target_image_path)
        # target_image = target_image.convert("L")  # Convert to grayscale
        target_array = np.array(target_image, dtype=np.uint16)
    else:
        # Create a blank image if it doesn't exist
        target_array = np.zeros(source_array.shape, dtype=np.uint16)

    # Make image zero everywhere but equal to trajectory_index for pixel = pixel_value
    if (
        pixel_value <= 0
    ):  # Assuming pixel_value > 0 means it is part of the object of interest
        coords, pixel_value = find_closest_nonzero(source_array, [y, x])

    target_array[source_array == pixel_value] = trajectory_index
    source_array[source_array == pixel_value] = 0

    # Save or update the image in the target directory for the frame
    updated_image = Image.fromarray(target_array)
    updated_image.save(r"{}".format(target_image_path))

    updated_s_image = Image.fromarray(source_array)
    updated_s_image.save(r"{}".format(temp_image_path))


def find_closest_nonzero(array, coords):
    """
    Find the closest nonzero pixel to a given pixel in an image.

    Args:
    image (numpy array): 2D numpy array representing the image.
    pixel (tuple): (x, y) coordinates of the target pixel.

    Returns:
    tuple: (x, y) coordinates of the closest non-zero pixel.
    """
    y, x = coords

    # Ensure the input image is a binary image where non-zero values are 1
    binary_array = np.where(array > 0, 1, 0)

    # Compute the Euclidean distance transform and indices of nearest non-zero pixels
    dist, indices = distance_transform_edt(1 - binary_array, return_indices=True)

    # The indices array contains the coordinates of the closest non-zero pixels
    closest_y, closest_x = indices[0][y, x], indices[1][y, x]

    # Retrieve the value of the closest non-zero pixel from the original image
    closest_value = array[closest_y, closest_x]

    if closest_value == 0:
        print("there is some issue")

    return (closest_y, closest_x), closest_value


def clean_up_temps(temp_dir):

    # Define the pattern for the files you want to delete
    file_pattern = os.path.join(temp_dir, "temp_mask*.tif")

    # Find all files that match the pattern
    temp_files = glob.glob(file_pattern)

    # Loop through and remove each file
    for temp_file in temp_files:
        os.remove(temp_file)
