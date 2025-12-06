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

# Curriculum learning configuration
# Progressive difficulty: start with n = k (NUM_EXAMPLES) and gradually increase by 1
# Edge weights also scale with n: max_weight = 10 + (n - 5) * 2 (from 10 at n=5 to 60 at n=30)
# Each stage uses a single node count, starting from k=5 up to 30
def get_scaled_max_weight(n: int) -> int:
    """Calculate max_weight that scales with graph size n."""
    # Linear scaling: start at 10 for n=5, increase by 2 per node, cap at 50
    return min(50, 10 + (n - 5) * 2)

CURRICULUM_STAGES = [
    {"num_nodes": [5], "samples": 3000, "name": "stage1_n5", "min_weight": 1, "max_weight": get_scaled_max_weight(5)},      # n=5: weights 1-10
    {"num_nodes": [6], "samples": 3000, "name": "stage2_n6", "min_weight": 1, "max_weight": get_scaled_max_weight(6)},      # n=6: weights 1-12
    {"num_nodes": [7], "samples": 3000, "name": "stage3_n7", "min_weight": 1, "max_weight": get_scaled_max_weight(7)},      # n=7: weights 1-14
    {"num_nodes": [8], "samples": 3000, "name": "stage4_n8", "min_weight": 1, "max_weight": get_scaled_max_weight(8)},      # n=8: weights 1-16
    {"num_nodes": [9], "samples": 3000, "name": "stage5_n9", "min_weight": 1, "max_weight": get_scaled_max_weight(9)},      # n=9: weights 1-18
    {"num_nodes": [10], "samples": 3000, "name": "stage6_n10", "min_weight": 1, "max_weight": get_scaled_max_weight(10)},   # n=10: weights 1-20
    {"num_nodes": [11], "samples": 3000, "name": "stage7_n11", "min_weight": 1, "max_weight": get_scaled_max_weight(11)},   # n=11: weights 1-22
    {"num_nodes": [12], "samples": 3000, "name": "stage8_n12", "min_weight": 1, "max_weight": get_scaled_max_weight(12)},  # n=12: weights 1-24
    {"num_nodes": [13], "samples": 3000, "name": "stage9_n13", "min_weight": 1, "max_weight": get_scaled_max_weight(13)},   # n=13: weights 1-26
    {"num_nodes": [14], "samples": 3000, "name": "stage10_n14", "min_weight": 1, "max_weight": get_scaled_max_weight(14)}, # n=14: weights 1-28
    {"num_nodes": [15], "samples": 3000, "name": "stage11_n15", "min_weight": 1, "max_weight": get_scaled_max_weight(15)}, # n=15: weights 1-30
    {"num_nodes": [16], "samples": 3000, "name": "stage12_n16", "min_weight": 1, "max_weight": get_scaled_max_weight(16)}, # n=16: weights 1-32
    {"num_nodes": [17], "samples": 3000, "name": "stage13_n17", "min_weight": 1, "max_weight": get_scaled_max_weight(17)}, # n=17: weights 1-34
    {"num_nodes": [18], "samples": 3000, "name": "stage14_n18", "min_weight": 1, "max_weight": get_scaled_max_weight(18)},  # n=18: weights 1-36
    {"num_nodes": [19], "samples": 3000, "name": "stage15_n19", "min_weight": 1, "max_weight": get_scaled_max_weight(19)}, # n=19: weights 1-38
    {"num_nodes": [20], "samples": 3000, "name": "stage16_n20", "min_weight": 1, "max_weight": get_scaled_max_weight(20)},  # n=20: weights 1-40
    {"num_nodes": [21], "samples": 3000, "name": "stage17_n21", "min_weight": 1, "max_weight": get_scaled_max_weight(21)}, # n=21: weights 1-42
    {"num_nodes": [22], "samples": 3000, "name": "stage18_n22", "min_weight": 1, "max_weight": get_scaled_max_weight(22)}, # n=22: weights 1-44
    {"num_nodes": [23], "samples": 3000, "name": "stage19_n23", "min_weight": 1, "max_weight": get_scaled_max_weight(23)}, # n=23: weights 1-46
    {"num_nodes": [24], "samples": 3000, "name": "stage20_n24", "min_weight": 1, "max_weight": get_scaled_max_weight(24)}, # n=24: weights 1-48
    {"num_nodes": [25], "samples": 3000, "name": "stage21_n25", "min_weight": 1, "max_weight": get_scaled_max_weight(25)},  # n=25: weights 1-50
    {"num_nodes": [26], "samples": 3000, "name": "stage22_n26", "min_weight": 1, "max_weight": get_scaled_max_weight(26)}, # n=26: weights 1-50 (capped)
    {"num_nodes": [27], "samples": 3000, "name": "stage23_n27", "min_weight": 1, "max_weight": get_scaled_max_weight(27)}, # n=27: weights 1-50 (capped)
    {"num_nodes": [28], "samples": 3000, "name": "stage24_n28", "min_weight": 1, "max_weight": get_scaled_max_weight(28)}, # n=28: weights 1-50 (capped)
    {"num_nodes": [29], "samples": 3000, "name": "stage25_n29", "min_weight": 1, "max_weight": get_scaled_max_weight(29)},  # n=29: weights 1-50 (capped)
    {"num_nodes": [30], "samples": 3000, "name": "stage26_n30", "min_weight": 1, "max_weight": get_scaled_max_weight(30)}, # n=30: weights 1-50 (capped)
]

# Dataset generation parameters
NUM_DATAPOINTS = sum(stage["samples"] for stage in CURRICULUM_STAGES)  # Total from curriculum stages
SEED = 182
NUM_NODES_LIST = list(range(1, 31))  # Graph sizes from 1 to 30 (N=30 max) - kept for backward compatibility
TRAIN_OUTPUT_FILE = "data/train_len.json"
VAL_OUTPUT_FILE = "data/val_len.json"
TRAIN_SPLIT = 0.1
USE_CURRICULUM = True  # Enable curriculum learning

# Graph generation parameters (Erdős–Rényi G(n, p))
EDGE_PROBABILITY = 0.3  # Probability p for G(n, p) distribution
MIN_WEIGHT = 1  # Minimum edge weight (natural number)
MAX_WEIGHT = 50  # Maximum edge weight (W = {n in N : n <= 50})

# In-context learning parameters
NUM_EXAMPLES = 5  # Number of example pairs (u_i, v_i) to include
MAX_REJECTION_ATTEMPTS = 100  # Maximum attempts for rejection sampling per pair
MAX_UNIQUE_ATTEMPTS = 500  # Maximum attempts to generate unique examples before regenerating graph

PRINT_PROGRESS_INTERVAL = 100  # Print progress every N datapoints

# Rehearsal/replay parameters
REHEARSAL_RATIO = 0.1  # Add 10% of stage samples as rehearsal from previous stages


def generate_erdos_renyi_graph(n: int, p: float, min_weight: int = MIN_WEIGHT, max_weight: int = MAX_WEIGHT) -> nx.Graph:
    """
    Generate a weighted undirected graph from Erdős–Rényi G(n, p) distribution.
    
    Args:
        n: Number of vertices
        p: Edge probability
        min_weight: Minimum edge weight (default: MIN_WEIGHT)
        max_weight: Maximum edge weight (default: MAX_WEIGHT)
    
    Returns:
        NetworkX undirected graph with random edge weights in [min_weight, max_weight]
    """
    # Generate Erdős–Rényi graph
    G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    
    # Add random weights to edges
    for u, v in G.edges():
        weight = random.randint(min_weight, max_weight)
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


def generate_datapoint(num_nodes: int, num_examples: int = NUM_EXAMPLES, min_weight: int = MIN_WEIGHT, max_weight: int = MAX_WEIGHT) -> Optional[dict]:
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
        min_weight: Minimum edge weight (default: MIN_WEIGHT)
        max_weight: Maximum edge weight (default: MAX_WEIGHT)
    
    Returns:
        Dictionary containing graph and shortest path length examples, or None if generation fails
    """
    # Try multiple times to generate a valid datapoint with unique examples
    for graph_attempt in range(MAX_UNIQUE_ATTEMPTS):
        # Generate Erdős–Rényi graph with scaled weights
        G = generate_erdos_renyi_graph(num_nodes, EDGE_PROBABILITY, min_weight, max_weight)
        
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
    Uses curriculum learning with progressive difficulty based on graph vertex count.
    Creates both train.json and val.json with no overlapping datapoints.
    """
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    
    if USE_CURRICULUM:
        return generate_curriculum_dataset()
    else:
        return generate_random_dataset()


def generate_curriculum_dataset():
    """
    Generate dataset using curriculum learning with progressive difficulty.
    Preserves order so easier examples (smaller graphs) come first.
    """
    all_datapoints = []
    seen_hashes = set()
    previous_stages_datapoints = []  # Track datapoints from previous stages for rehearsal
    
    print(f"Generating curriculum learning dataset with {len(CURRICULUM_STAGES)} stages...")
    print(f"Graph parameters: Erdős–Rényi G(n, p) with n <= 30, p = {EDGE_PROBABILITY}")
    print(f"Edge weights: scaling with graph size (from 1-10 at n=5 to 1-50 at n=30)")
    print(f"In-context learning: {NUM_EXAMPLES} examples + 1 query per datapoint")
    print(f"Rehearsal ratio: {REHEARSAL_RATIO*100:.1f}% (adding {REHEARSAL_RATIO*100:.1f}% rehearsal samples from previous stages)")
    print(f"Total target samples: {NUM_DATAPOINTS}\n")
    
    # Generate data for each curriculum stage
    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        stage_name = stage["name"]
        num_nodes_list = stage["num_nodes"]
        target_samples = stage["samples"]
        stage_min_weight = stage.get("min_weight", MIN_WEIGHT)
        stage_max_weight = stage.get("max_weight", MAX_WEIGHT)
        
        print(f"{'='*80}")
        print(f"STAGE {stage_idx + 1}: {stage_name}")
        print(f"  Nodes: {num_nodes_list}")
        print(f"  Edge weights: [{stage_min_weight}, {stage_max_weight}]")
        print(f"  Target samples: {target_samples}")
        print(f"{'='*80}")
        
        stage_datapoints = []
        attempts = 0
        max_attempts = target_samples * 20  # Prevent infinite loops
        
        while len(stage_datapoints) < target_samples and attempts < max_attempts:
            attempts += 1
            
            # Select number of nodes from this stage's range
            num_nodes = random.choice(num_nodes_list)
            datapoint = generate_datapoint(num_nodes, NUM_EXAMPLES, stage_min_weight, stage_max_weight)
            
            # Skip if generation failed
            if datapoint is None:
                continue
            
            # Check for duplicates
            datapoint_hash = get_datapoint_hash(datapoint)
            if datapoint_hash not in seen_hashes:
                seen_hashes.add(datapoint_hash)
                datapoint['id'] = len(all_datapoints)
                datapoint['stage_id'] = len(stage_datapoints)
                datapoint['curriculum_stage'] = stage_name
                stage_datapoints.append(datapoint)
                all_datapoints.append(datapoint)
                
                if len(stage_datapoints) % PRINT_PROGRESS_INTERVAL == 0:
                    print(f"  Generated {len(stage_datapoints)}/{target_samples} samples...")
        
        if len(stage_datapoints) < target_samples:
            print(f"  Warning: Only generated {len(stage_datapoints)}/{target_samples} samples for this stage.")
        
        # Add rehearsal samples from previous stages (if not first stage)
        rehearsal_samples = []
        if stage_idx > 0 and len(previous_stages_datapoints) > 0:
            num_rehearsal = int(target_samples * REHEARSAL_RATIO)
            if num_rehearsal > 0:
                # Sample random examples from previous stages
                rehearsal_candidates = random.sample(previous_stages_datapoints, min(num_rehearsal, len(previous_stages_datapoints)))
                for rehearsal_dp in rehearsal_candidates:
                    # Create a copy with updated metadata
                    rehearsal_copy = rehearsal_dp.copy()
                    rehearsal_copy['id'] = len(all_datapoints) + len(rehearsal_samples)
                    rehearsal_copy['stage_id'] = len(stage_datapoints) + len(rehearsal_samples)
                    rehearsal_copy['curriculum_stage'] = stage_name
                    rehearsal_copy['is_rehearsal'] = True
                    rehearsal_copy['original_stage'] = rehearsal_dp.get('curriculum_stage', 'unknown')
                    rehearsal_samples.append(rehearsal_copy)
                
                print(f"  Added {len(rehearsal_samples)} rehearsal samples from previous stages")
        
        # Combine stage samples and rehearsal samples
        stage_datapoints.extend(rehearsal_samples)
        all_datapoints.extend(rehearsal_samples)
        
        # Update previous_stages_datapoints for next stage's rehearsal
        previous_stages_datapoints.extend([dp for dp in stage_datapoints if not dp.get('is_rehearsal', False)])
        
        print(f"  ✓ Completed stage {stage_idx + 1}: {len(stage_datapoints)} total samples ({len(stage_datapoints) - len(rehearsal_samples)} new + {len(rehearsal_samples)} rehearsal)\n")
    
    total_samples = len(all_datapoints)
    print(f"\n{'='*80}")
    print(f"TOTAL: Generated {total_samples} unique datapoints")
    print(f"{'='*80}\n")
    
    # Split into train and val
    # Training set: preserve curriculum order (easier examples first)
    # Validation set: randomly sample from entire dataset
    num_val_samples = int(len(all_datapoints) * (1 - TRAIN_SPLIT))
    num_train_samples = len(all_datapoints) - num_val_samples
    
    # Randomly sample validation set from entire dataset
    val_indices = set(random.sample(range(len(all_datapoints)), num_val_samples))
    train_datapoints = [dp for idx, dp in enumerate(all_datapoints) if idx not in val_indices]
    val_datapoints = [all_datapoints[idx] for idx in val_indices]
    
    # Shuffle validation set (it's already random, but shuffle for good measure)
    random.shuffle(val_datapoints)
    
    # Update IDs to be sequential within each split
    for idx, dp in enumerate(train_datapoints):
        dp['id'] = idx
    for idx, dp in enumerate(val_datapoints):
        dp['id'] = idx
    
    # Convert to instruction format
    train_dataset = convert_to_qwen_format(train_datapoints)
    val_dataset = convert_to_qwen_format(val_datapoints)
    
    # Save train dataset (preserve order for curriculum learning!)
    train_path = Path(__file__).parent / TRAIN_OUTPUT_FILE
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_dataset, f, separators=(',', ':'))
    
    # Save val dataset
    val_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_dataset, f, separators=(',', ':'))
    
    # Print stage distribution
    print(f"Stage distribution in training set:")
    stage_counts = {}
    for dp in train_datapoints:
        stage = dp.get('curriculum_stage', 'unknown')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    for stage in CURRICULUM_STAGES:
        stage_name = stage["name"]
        count = stage_counts.get(stage_name, 0)
        percentage = 100 * count / len(train_datapoints) if train_datapoints else 0
        print(f"  {stage_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✓ Train dataset: {train_path} ({len(train_datapoints)} datapoints)")
    print(f"✓ Val dataset: {val_path} ({len(val_datapoints)} datapoints)")
    print(f"✓ Split: {100*TRAIN_SPLIT:.1f}% train / {100*(1-TRAIN_SPLIT):.1f}% val")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: Training data is in CURRICULUM ORDER!")
    print("Easy examples (smaller graphs) come first, harder examples (larger graphs) come later.")
    print("Do NOT shuffle the training data!")
    print("\nValidation set is RANDOMLY sampled from the entire dataset.")
    print(f"{'='*80}\n")
    
    return train_datapoints, val_datapoints


def generate_random_dataset():
    """
    Generate dataset with random node selection (non-curriculum, for backward compatibility).
    """
    datapoints = []
    seen_hashes = set()
    attempts = 0
    max_attempts = NUM_DATAPOINTS * 20  # Prevent infinite loops
    
    print(f"Generating {NUM_DATAPOINTS} unique datapoints (random sampling)...")
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
