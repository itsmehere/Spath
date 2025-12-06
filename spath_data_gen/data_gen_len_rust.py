"""
Fast Rust-based implementation of shortest path data generation.
This module uses Rust for graph operations and shortest path calculations.
"""
import json
import random
import numpy as np
from typing import List, Optional
import hashlib
from pathlib import Path

try:
    from spath_data_gen_rust import generate_datapoint_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("Warning: Rust module not available. Install with: cd spath_data_gen_rust && maturin develop")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Dataset generation parameters
NUM_DATAPOINTS = 500000
SEED = 182
NUM_NODES_LIST = list(range(1, 31))  # Graph sizes from 1 to 30 (N=30 max)
TRAIN_OUTPUT_FILE = "data/train_len.json"
VAL_OUTPUT_FILE = "data/val_len.json"
TRAIN_SPLIT = 0.999

# Graph generation parameters (Erdős–Rényi G(n, p))
EDGE_PROBABILITY = 0.3  # Probability p for G(n, p) distribution
MIN_WEIGHT = 1  # Minimum edge weight (natural number)
MAX_WEIGHT = 50  # Maximum edge weight (W = {n in N : n <= 50})

# In-context learning parameters
NUM_EXAMPLES = 5  # Number of example pairs (u_i, v_i) to include
MAX_REJECTION_ATTEMPTS = 100  # Maximum attempts for rejection sampling per pair
MAX_UNIQUE_ATTEMPTS = 500  # Maximum attempts to generate unique examples before regenerating graph

PRINT_PROGRESS_INTERVAL = 100  # Print progress every N datapoints


def generate_datapoint(num_nodes: int, num_examples: int = NUM_EXAMPLES) -> Optional[dict]:
    """
    Generate a single datapoint using Rust implementation.
    
    Args:
        num_nodes: Number of nodes in the graph (n <= 30)
        num_examples: Number of example pairs to include (k)
    
    Returns:
        Dictionary containing graph and shortest path length examples, or None if generation fails
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust module not available. Please build it first.")
    
    # Generate random seed for this attempt
    seed = random.getrandbits(64)
    
    result = generate_datapoint_rust(
        num_nodes=num_nodes,
        edge_probability=EDGE_PROBABILITY,
        min_weight=MIN_WEIGHT,
        max_weight=MAX_WEIGHT,
        num_examples=num_examples,
        max_rejection_attempts=MAX_REJECTION_ATTEMPTS,
        max_unique_attempts=MAX_UNIQUE_ATTEMPTS,
        seed=seed,
    )
    
    if result is None:
        return None
    
    # Convert PyDict to Python dict
    # The Rust code returns a PyDict which is dict-like in Python
    examples_list = result["examples"]
    return {
        "adjacency_matrix": [list(row) for row in result["adjacency_matrix"]],
        "num_nodes": result["num_nodes"],
        "examples": [
            {
                "u": ex["u"],
                "v": ex["v"],
                "length": ex["length"]
            }
            for ex in examples_list
        ],
        "query": {
            "u": result["query"]["u"],
            "v": result["query"]["v"],
            "length": result["query"]["length"]
        }
    }


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
    if not RUST_AVAILABLE:
        raise ImportError("Rust module not available. Please build it first with: cd spath_data_gen_rust && maturin develop")
    
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    datapoints = []
    seen_hashes = set()
    attempts = 0
    max_attempts = NUM_DATAPOINTS * 20  # Prevent infinite loops
    
    print(f"Generating {NUM_DATAPOINTS} unique datapoints using Rust implementation...")
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

