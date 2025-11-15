import json
import random
import numpy as np
from typing import List, Tuple, Optional
import heapq
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset generation parameters
NUM_DATAPOINTS = 500000
SEED = 182
NUM_NODES_LIST = [10,11,12,13,14,15]  # Graph sizes (e.g., [1,2] for 1x1 or 2x2 matrices)
TRAIN_OUTPUT_FILE = "data/train.json"
VAL_OUTPUT_FILE = "data/val.json"
TRAIN_SPLIT = 0.999

# Graph generation parameters
EDGE_PROBABILITY = 0.3  # Probability of an edge existing between any two nodes
MIN_WEIGHT = 1  # Minimum edge weight
MAX_WEIGHT = 10  # Maximum edge weight

PRINT_PROGRESS_INTERVAL = 10  # Print progress every N datapoints


def dijkstra_all_reachable(adjacency_matrix: np.ndarray, start: int) -> List[int]:
    """
    Find all nodes reachable from start using Dijkstra's algorithm.
    
    Args:
        adjacency_matrix: NxN adjacency matrix (0 means no edge, >0 means edge weight)
        start: Starting node index (0-indexed)
    
    Returns:
        List of node indices reachable from start (including start itself)
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    visited = [False] * n
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    reachable = []
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        reachable.append(current)
        
        # Check all neighbors
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return reachable


def dijkstra(adjacency_matrix: np.ndarray, start: int, end: int) -> Optional[List[int]]:
    """
    Compute shortest path from start to end using Dijkstra's algorithm.
    
    Args:
        adjacency_matrix: NxN adjacency matrix (0 means no edge, >0 means edge weight)
        start: Starting node index (0-indexed)
        end: Ending node index (0-indexed)
    
    Returns:
        List of node indices representing the shortest path, or None if no path exists
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = previous[node]
            return path[::-1]  # Reverse to get path from start to end
        
        # Check all neighbors
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return None  # No path found


def generate_random_graph(num_nodes: int) -> np.ndarray:
    """
    Generate a random directed graph as an adjacency matrix.
    
    Args:
        num_nodes: Number of nodes in the graph
    
    Returns:
        NxN adjacency matrix (0 means no edge, >0 means edge weight)
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < EDGE_PROBABILITY:
                adjacency_matrix[i][j] = random.randint(MIN_WEIGHT, MAX_WEIGHT)
    
    return adjacency_matrix


def get_datapoint_hash(datapoint: dict) -> str:
    """
    Create a hash for a datapoint to check for duplicates.
    Uses adjacency matrix, start node, and end node.
    
    Args:
        datapoint: Dictionary containing adjacency_matrix, start_node, end_node
    
    Returns:
        String hash of the datapoint
    """
    # Convert adjacency matrix to a tuple for hashing
    adj_tuple = tuple(tuple(row) for row in datapoint['adjacency_matrix'])
    # Create a unique identifier
    unique_str = f"{adj_tuple}_{datapoint['start_node']}_{datapoint['end_node']}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def generate_datapoint(num_nodes: int) -> dict:
    """
    Generate a single datapoint with adjacency matrix and shortest path.
    
    Args:
        num_nodes: Number of nodes in the graph (should be <= 10)
    
    Returns:
        Dictionary containing adjacency matrix and shortest path
    """
    # Generate random graph
    adjacency_matrix = generate_random_graph(num_nodes)
    
    # Pick a random start node
    start_node = random.randint(0, num_nodes - 1)
    
    # Find all reachable nodes from start
    reachable_nodes = dijkstra_all_reachable(adjacency_matrix, start_node)
    
    # Remove start node from reachable nodes (we want a path to a different node)
    reachable_end_nodes = [node for node in reachable_nodes if node != start_node]
    
    # If no other nodes are reachable, skip this datapoint and try again
    # (This should be rare with reasonable edge probability)
    if not reachable_end_nodes:
        # Try generating a new graph
        return generate_datapoint(num_nodes)
    
    # Pick a random end node from reachable nodes
    end_node = random.choice(reachable_end_nodes)
    
    # Compute shortest path
    shortest_path = dijkstra(adjacency_matrix, start_node, end_node)
    
    # Convert to 1-indexed for output
    shortest_path_1_indexed = [node + 1 for node in shortest_path] if shortest_path else None
    
    return {
        "adjacency_matrix": adjacency_matrix.tolist(),
        "shortest_path": shortest_path_1_indexed,
        "num_nodes": num_nodes,
        "start_node": start_node + 1,  # Convert to 1-indexed
        "end_node": end_node + 1  # Convert to 1-indexed
    }


def convert_to_qwen_format(datapoints: List[dict]) -> List[dict]:
    """
    Convert datapoints to instruction-input-output format.
    
    Args:
        datapoints: List of datapoint dictionaries
    
    Returns:
        List of instruction format dictionaries
    """
    dataset = []
    for datapoint in datapoints:
        # Format adjacency matrix as a string for the prompt (single line)
        adj_matrix_str = json.dumps(datapoint['adjacency_matrix'])
        
        input_text = f"Here is an adjacency matrix representing a directed graph: {adj_matrix_str}\nThe adjacency matrix is {datapoint['num_nodes']}x{datapoint['num_nodes']}, where 0 means no edge and a positive number represents the edge weight."
        instruction = f"What is the shortest path from node {datapoint['start_node']} to node {datapoint['end_node']}?"
        
        # Create output with the shortest path as a list
        shortest_path_list = json.dumps(datapoint['shortest_path'])
        output_text = f"Shortest path from node {datapoint['start_node']} to node {datapoint['end_node']} is: {shortest_path_list}"
        
        # Build instruction format datapoint
        formatted_datapoint = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        dataset.append(formatted_datapoint)
    
    return dataset


def generate_dataset():
    """
    Generate a dataset of synthetic shortest path graph data.
    Uses parameters defined at the top of the file.
    Creates both train.json and val.json with no overlapping datapoints.
    """
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    datapoints = []
    seen_hashes = set()
    attempts = 0
    max_attempts = NUM_DATAPOINTS * 10  # Prevent infinite loops
    
    print(f"Generating {NUM_DATAPOINTS} unique datapoints...")
    
    while len(datapoints) < NUM_DATAPOINTS and attempts < max_attempts:
        attempts += 1
        
        # Randomly select number of nodes from NUM_NODES_LIST
        num_nodes = random.choice(NUM_NODES_LIST)
        datapoint = generate_datapoint(num_nodes)
        
        # Check for duplicates
        datapoint_hash = get_datapoint_hash(datapoint)
        if datapoint_hash not in seen_hashes:
            seen_hashes.add(datapoint_hash)
            datapoint['id'] = len(datapoints)
            datapoints.append(datapoint)
            
            if len(datapoints) % PRINT_PROGRESS_INTERVAL == 0:
                print(f"Generated {len(datapoints)}/{NUM_DATAPOINTS} unique datapoints...")
        else:
            # Duplicate found, skip it
            continue
    
    if len(datapoints) < NUM_DATAPOINTS:
        print(f"Warning: Only generated {len(datapoints)} unique datapoints out of {NUM_DATAPOINTS} requested.")
    
    # Shuffle datapoints before splitting
    random.shuffle(datapoints)
    
    # Split into train and val
    split_idx = int(len(datapoints) * TRAIN_SPLIT)
    train_datapoints = datapoints[:split_idx]
    val_datapoints = datapoints[split_idx:]
    
    # Update IDs to be sequential within each split
    for idx, dp in enumerate(train_datapoints):
        dp['id'] = idx
    for idx, dp in enumerate(val_datapoints):
        dp['id'] = idx
    
    # Convert to instruction format
    train_dataset = convert_to_qwen_format(train_datapoints)
    val_dataset = convert_to_qwen_format(val_datapoints)
    
    # Save train dataset
    train_path = Path(__file__).parent / TRAIN_OUTPUT_FILE
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_dataset, f, separators=(',', ':'))
    
    # Save val dataset
    val_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_dataset, f, separators=(',', ':'))
    
    print(f"Train dataset saved to {train_path} ({len(train_datapoints)} datapoints)")
    print(f"Val dataset saved to {val_path} ({len(val_datapoints)} datapoints)")
    print(f"Total unique datapoints: {len(datapoints)}")
    print(f"Train/Val split: {len(train_datapoints)}/{len(val_datapoints)} ({100*TRAIN_SPLIT:.1f}%/{100*(1-TRAIN_SPLIT):.1f}%)")
    
    return train_datapoints, val_datapoints


if __name__ == "__main__":
    generate_dataset()

