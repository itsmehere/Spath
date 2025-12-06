import networkx as nx
import json
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset generation parameters
NUM_DATAPOINTS = 500000
SEED = 182
NUM_NODES_LIST = list(range(1, 31))  # Graph sizes from 1 to 30 (N=30 max)
TRAIN_OUTPUT_FILE = "data/train_len.json"
VAL_OUTPUT_FILE = "data/val_len.json"
TRAIN_SPLIT = 0.1

# Graph generation parameters (Erdős–Rényi G(n, p))
EDGE_PROBABILITY = 0.3  # Probability p for G(n, p) distribution
MIN_WEIGHT = 1  # Minimum edge weight (natural number)
MAX_WEIGHT = 50  # Maximum edge weight (W = {n in N : n <= 50})

# In-context learning parameters
NUM_EXAMPLES = 5  # Number of example pairs (u_i, v_i) to include
MAX_REJECTION_ATTEMPTS = 100  # Maximum attempts for rejection sampling per pair
MAX_UNIQUE_ATTEMPTS = 500  # Maximum attempts to generate unique examples before regenerating graph

PRINT_PROGRESS_INTERVAL = 100  # Print progress every N datapoints


def generate_erdos_renyi_graph(n: int, p: float) -> nx.Graph:
    """
    Generate a weighted undirected graph from Erdős–Rényi G(n, p) distribution.
    
    Args:
        n: Number of vertices
        p: Edge probability
    
    Returns:
        NetworkX undirected graph with random edge weights in [MIN_WEIGHT, MAX_WEIGHT]
    """
    # Generate Erdős–Rényi graph
    G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    
    # Add random weights to edges
    for u, v in G.edges():
        weight = random.randint(MIN_WEIGHT, MAX_WEIGHT)
        G[u][v]['weight'] = weight
    
    return G


def shortest_path_length(G: nx.Graph, u: int, v: int) -> Optional[int]:
    """
    Compute the length of the shortest path between vertices u and v.
    
    Args:
        G: NetworkX weighted undirected graph
        u: Source vertex (0-indexed)
        v: Target vertex (0-indexed)
    
    Returns:
        Length of shortest path, or None if no path exists
    """
    try:
        length = nx.shortest_path_length(G, u, v, weight='weight')
        return int(length)
    except nx.NetworkXNoPath:
        return None


def sample_vertex_pair_with_path(G: nx.Graph, max_attempts: int = MAX_REJECTION_ATTEMPTS) -> Optional[Tuple[int, int, int]]:
    """
    Sample a vertex pair (u, v) and compute shortest path length.
    Uses rejection sampling to guarantee a path exists.
    
    Args:
        G: NetworkX weighted undirected graph
        max_attempts: Maximum number of attempts for rejection sampling
    
    Returns:
        Tuple (u, v, length) if path exists, None otherwise
    """
    n = G.number_of_nodes()
    if n < 2:
        return None
    
    for _ in range(max_attempts):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        
        if u == v:
            continue
        
        length = shortest_path_length(G, u, v)
        if length is not None:
            return (u, v, length)
    
    return None


def graph_to_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix representation.
    
    Args:
        G: NetworkX weighted undirected graph
    
    Returns:
        NxN adjacency matrix (0 means no edge, >0 means edge weight)
    """
    n = G.number_of_nodes()
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        adj_matrix[u][v] = weight
        adj_matrix[v][u] = weight  # Undirected graph
    
    return adj_matrix


def generate_datapoint(num_nodes: int, num_examples: int = NUM_EXAMPLES) -> Optional[dict]:
    """
    Generate a single datapoint with graph and shortest path length examples.
    
    According to the problem formulation:
    - Generate graph from Erdős–Rényi G(n, p) distribution
    - Sample k vertex pairs (u_i, v_i) and calculate shortest path lengths
    - Use rejection sampling to guarantee paths exist
    - Ensure all example pairs are unique (no repeats)
    - Append a final query pair (u_q, v_q) for the model to complete
    - If unable to generate unique examples, regenerate the graph
    
    Args:
        num_nodes: Number of nodes in the graph (n <= 30)
        num_examples: Number of example pairs to include (k)
    
    Returns:
        Dictionary containing graph and shortest path length examples, or None if generation fails
    """
    # Try multiple times to generate a valid datapoint with unique examples
    for graph_attempt in range(MAX_UNIQUE_ATTEMPTS):
        # Generate Erdős–Rényi graph
        G = generate_erdos_renyi_graph(num_nodes, EDGE_PROBABILITY)
        
        # Convert to adjacency matrix for representation
        adjacency_matrix = graph_to_adjacency_matrix(G)
        
        # Track used pairs to ensure uniqueness (using frozenset for undirected pairs)
        used_pairs = set()
        
        # Sample k example pairs with rejection sampling, ensuring uniqueness
        examples = []
        example_attempts = 0
        
        while len(examples) < num_examples and example_attempts < MAX_UNIQUE_ATTEMPTS:
            example_attempts += 1
            result = sample_vertex_pair_with_path(G, max_attempts=MAX_REJECTION_ATTEMPTS)
            
            if result is None:
                # Can't find a valid pair, break and try new graph
                break
            
            u, v, length = result
            # Use frozenset to handle undirected pairs (u,v) = (v,u)
            pair_key = frozenset({u, v})
            
            # Check if this pair is already used
            if pair_key in used_pairs:
                continue  # Try again for a different pair
            
            # Add to used pairs and examples
            used_pairs.add(pair_key)
            examples.append({
                'u': u + 1,  # Convert to 1-indexed
                'v': v + 1,  # Convert to 1-indexed
                'length': length
            })
        
        # If we couldn't generate enough unique examples, try a new graph
        if len(examples) < num_examples:
            continue
        
        # Sample final query pair (u_q, v_q), ensuring it's different from all examples
        query_attempts = 0
        query_result = None
        
        while query_attempts < MAX_UNIQUE_ATTEMPTS:
            query_attempts += 1
            result = sample_vertex_pair_with_path(G, max_attempts=MAX_REJECTION_ATTEMPTS)
            
            if result is None:
                # Can't find a valid query pair, try new graph
                break
            
            u_q, v_q, query_length = result
            query_pair_key = frozenset({u_q, v_q})
            
            # Ensure query pair is different from all examples
            if query_pair_key not in used_pairs:
                query_result = (u_q, v_q, query_length)
                break
        
        # If we couldn't find a unique query pair, try a new graph
        if query_result is None:
            continue
        
        u_q, v_q, query_length = query_result
        
        # Successfully generated datapoint with unique examples and query
        return {
            "adjacency_matrix": adjacency_matrix.tolist(),
            "num_nodes": num_nodes,
            "examples": examples,  # List of k example pairs with their shortest path lengths
            "query": {
                "u": u_q + 1,  # Convert to 1-indexed
                "v": v_q + 1,  # Convert to 1-indexed
                "length": query_length  # Ground truth answer
            }
        }
    
    # Failed to generate unique examples after many attempts
    return None


def get_datapoint_hash(datapoint: dict) -> str:
    """
    Create a hash for a datapoint to check for duplicates.
    
    Args:
        datapoint: Dictionary containing adjacency_matrix, examples, and query
    
    Returns:
        String hash of the datapoint
    """
    # Convert adjacency matrix to a tuple for hashing
    adj_tuple = tuple(tuple(row) for row in datapoint['adjacency_matrix'])
    # Hash examples and query
    examples_str = json.dumps(datapoint['examples'], sort_keys=True)
    query_str = json.dumps(datapoint['query'], sort_keys=True)
    unique_str = f"{adj_tuple}_{examples_str}_{query_str}"
    return hashlib.md5(unique_str.encode()).hexdigest()


def convert_to_qwen_format(datapoints: List[dict]) -> List[dict]:
    """
    Convert datapoints to instruction-input-output format for in-context learning.
    Uses the format from example_prompt.txt:
    - Header: "Given the following undirected weighted graph:"
    - Vertices section with space-separated vertex numbers
    - Edge Weights section with "u v: weight" format
    - Task description
    - Shortest path length section with "Example:" blocks containing "Input:" and "Output:" format
    
    Args:
        datapoints: List of datapoint dictionaries
    
    Returns:
        List of instruction format dictionaries
    """
    dataset = []
    for datapoint in datapoints:
        adj_matrix = np.array(datapoint['adjacency_matrix'])
        num_nodes = datapoint['num_nodes']
        
        # Extract vertices list
        vertices = list(range(1, num_nodes + 1))  # 1-indexed
        vertices_str = " ".join(str(v) for v in vertices)
        
        # Extract edges (for undirected graph, only list each edge once: u < v)
        edges = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):  # Only upper triangle
                weight = adj_matrix[u][v]
                if weight > 0:  # Edge exists
                    edges.append((u + 1, v + 1, weight))  # Convert to 1-indexed
        
        # Sort edges for consistent output
        edges.sort()
        
        # Build graph section matching example_prompt.txt format
        graph_parts = [
            "Given the following undirected weighted graph:",
            "",
            "Vertices:",
            vertices_str,
            "",
            "Edge Weights:"
        ]
        
        for u, v, weight in edges:
            graph_parts.append(f"{u} {v}: {weight}")
        
        # Add task description
        graph_parts.append("")
        graph_parts.append("Task: For each pair u v, output the length of the shortest path between u and v.")
        
        # Add "Shortest path length:" header
        graph_parts.append("")
        graph_parts.append("Shortest path length:")
        
        # Add example pairs in "Example:" format with "Input:" and "Output:"
        for example in datapoint['examples']:
            graph_parts.append("Example:")
            graph_parts.append(f"Input: {example['u']} {example['v']}")
            graph_parts.append(f"Output: {example['length']}")
            graph_parts.append("")
        
        # Add query pair in "Input:" and "Output: " format (without answer)
        query = datapoint['query']
        graph_parts.append(f"Input: {query['u']} {query['v']}")
        graph_parts.append("Output: ")
        
        input_text = "\n".join(graph_parts)
        
        # Output is just the shortest path length
        output_text = str(query['length'])
        
        # Build datapoint with just input and output (matching example_prompt.txt style)
        formatted_datapoint = {
            "input": input_text,
            "output": output_text
        }
        dataset.append(formatted_datapoint)
    
    return dataset


def generate_dataset():
    """
    Generate a dataset of synthetic shortest path length graph data.
    Uses parameters defined at the top of the file.
    Creates both train.json and val.json with no overlapping datapoints.
    """
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    datapoints = []
    seen_hashes = set()
    attempts = 0
    max_attempts = NUM_DATAPOINTS * 20  # Prevent infinite loops
    
    print(f"Generating {NUM_DATAPOINTS} unique datapoints...")
    print(f"Graph parameters: Erdős–Rényi G(n, p) with n <= 30, p = {EDGE_PROBABILITY}")
    print(f"Edge weights: integers in [{MIN_WEIGHT}, {MAX_WEIGHT}]")
    print(f"In-context learning: {NUM_EXAMPLES} examples + 1 query per datapoint")
    
    while len(datapoints) < NUM_DATAPOINTS and attempts < max_attempts:
        attempts += 1
        
        # Randomly select number of nodes from NUM_NODES_LIST
        num_nodes = random.choice(NUM_NODES_LIST)
        datapoint = generate_datapoint(num_nodes, NUM_EXAMPLES)
        
        # Skip if generation failed
        if datapoint is None:
            continue
        
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
