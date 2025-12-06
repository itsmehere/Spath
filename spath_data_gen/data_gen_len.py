import networkx as nx
import json
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import hashlib
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Default configuration parameters (used when not provided as arguments)
DEFAULT_SEED = 182
DEFAULT_TRAIN_OUTPUT_FILE = "data/train_len.json"
DEFAULT_VAL_OUTPUT_FILE = "data/val_len.json"
DEFAULT_TRAIN_SPLIT = 0.1
DEFAULT_USE_CURRICULUM = True
DEFAULT_EDGE_PROBABILITY = 0.3
DEFAULT_MIN_WEIGHT = 1
DEFAULT_MAX_WEIGHT = 50
DEFAULT_NUM_EXAMPLES = 5
DEFAULT_MAX_REJECTION_ATTEMPTS = 100
DEFAULT_MAX_UNIQUE_ATTEMPTS = 500
DEFAULT_PRINT_PROGRESS_INTERVAL = 100
DEFAULT_REHEARSAL_RATIO = 0.1
DEFAULT_SAMPLES_PER_STAGE = 3000
DEFAULT_MIN_NODES = 5
DEFAULT_MAX_NODES = 30
DEFAULT_NUM_DATAPOINTS = 500000

def get_scaled_max_weight(n: int, max_weight_cap: int = 50) -> int:
    """Calculate max_weight that scales with graph size n."""
    # Linear scaling: start at 10 for n=5, increase by 2 per node, cap at max_weight_cap
    return min(max_weight_cap, 10 + (n - 5) * 2)

def build_curriculum_stages(
    min_nodes: int = DEFAULT_MIN_NODES,
    max_nodes: int = DEFAULT_MAX_NODES,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    samples_per_stage: int = DEFAULT_SAMPLES_PER_STAGE,
    min_weight: int = DEFAULT_MIN_WEIGHT,
    max_weight: int = DEFAULT_MAX_WEIGHT,
) -> List[Dict]:
    """
    Build curriculum stages dynamically based on parameters.
    
    Args:
        min_nodes: Minimum number of nodes (starting point, typically = num_examples)
        max_nodes: Maximum number of nodes (ending point)
        num_examples: Number of examples (k), used as starting point if min_nodes not specified
        samples_per_stage: Number of samples per curriculum stage
        min_weight: Minimum edge weight
        max_weight: Maximum edge weight (cap for scaling)
    
    Returns:
        List of curriculum stage dictionaries
    """
    stages = []
    start_node = max(min_nodes, num_examples)  # Start from max(min_nodes, k)
    
    stage_num = 1
    for n in range(start_node, max_nodes + 1):
        stages.append({
            "num_nodes": [n],
            "samples": samples_per_stage,
            "name": f"stage{stage_num}_n{n}",
            "min_weight": min_weight,
            "max_weight": get_scaled_max_weight(n, max_weight),
        })
        stage_num += 1
    
    return stages


def generate_erdos_renyi_graph(n: int, p: float, min_weight: int = DEFAULT_MIN_WEIGHT, max_weight: int = DEFAULT_MAX_WEIGHT) -> nx.Graph:
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


def sample_vertex_pair_with_path(G: nx.Graph, max_attempts: int = DEFAULT_MAX_REJECTION_ATTEMPTS) -> Optional[Tuple[int, int, int]]:
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


def generate_datapoint(
    num_nodes: int,
    edge_probability: float,
    num_examples: int,
    min_weight: int,
    max_weight: int,
    max_rejection_attempts: int,
    max_unique_attempts: int,
) -> Optional[dict]:
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
    for graph_attempt in range(max_unique_attempts):
        # Generate Erdős–Rényi graph with scaled weights
        G = generate_erdos_renyi_graph(num_nodes, edge_probability, min_weight, max_weight)
        
        # Convert to adjacency matrix for representation
        adjacency_matrix = graph_to_adjacency_matrix(G)
        
        # Track used pairs to ensure uniqueness (using frozenset for undirected pairs)
        used_pairs = set()
        
        # Sample k example pairs with rejection sampling, ensuring uniqueness
        examples = []
        example_attempts = 0
        
        while len(examples) < num_examples and example_attempts < max_unique_attempts:
            example_attempts += 1
            result = sample_vertex_pair_with_path(G, max_attempts=max_rejection_attempts)
            
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
        
        while query_attempts < max_unique_attempts:
            query_attempts += 1
            result = sample_vertex_pair_with_path(G, max_attempts=max_rejection_attempts)
            
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


def generate_dataset(
    seed: int = 182,
    train_output_file: str = "data/train_len.json",
    val_output_file: str = "data/val_len.json",
    train_split: float = 0.1,
    use_curriculum: bool = True,
    edge_probability: float = 0.3,
    min_weight: int = 1,
    max_weight: int = 50,
    num_examples: int = 5,
    max_rejection_attempts: int = 100,
    max_unique_attempts: int = 500,
    print_progress_interval: int = 100,
    rehearsal_ratio: float = 0.1,
    samples_per_stage: int = 3000,
    min_nodes: int = 5,
    max_nodes: int = 30,
    num_datapoints: Optional[int] = None,
):
    """
    Generate a dataset of synthetic shortest path length graph data.
    Uses curriculum learning with progressive difficulty based on graph vertex count.
    Creates both train.json and val.json with no overlapping datapoints.
    
    Args:
        seed: Random seed for reproducibility
        train_output_file: Path to output training dataset file
        val_output_file: Path to output validation dataset file
        train_split: Fraction of data for training (remainder goes to validation)
        use_curriculum: Whether to use curriculum learning (True) or random sampling (False)
        edge_probability: Probability p for Erdős–Rényi G(n, p) distribution
        min_weight: Minimum edge weight
        max_weight: Maximum edge weight
        num_examples: Number of example pairs (u_i, v_i) to include
        max_rejection_attempts: Maximum attempts for rejection sampling per pair
        max_unique_attempts: Maximum attempts to generate unique examples before regenerating graph
        print_progress_interval: Print progress every N datapoints
        rehearsal_ratio: Ratio of rehearsal samples from previous stages (0.1 = 10%)
        samples_per_stage: Number of samples per curriculum stage
        min_nodes: Minimum number of nodes (starting point for curriculum)
        max_nodes: Maximum number of nodes (ending point for curriculum)
        num_datapoints: Total number of datapoints (only used if use_curriculum=False)
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    if use_curriculum:
        return generate_curriculum_dataset(
            seed=seed,
            train_output_file=train_output_file,
            val_output_file=val_output_file,
            train_split=train_split,
            edge_probability=edge_probability,
            min_weight=min_weight,
            max_weight=max_weight,
            num_examples=num_examples,
            max_rejection_attempts=max_rejection_attempts,
            max_unique_attempts=max_unique_attempts,
            print_progress_interval=print_progress_interval,
            rehearsal_ratio=rehearsal_ratio,
            samples_per_stage=samples_per_stage,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
    else:
        return generate_random_dataset(
            seed=seed,
            train_output_file=train_output_file,
            val_output_file=val_output_file,
            train_split=train_split,
            edge_probability=edge_probability,
            min_weight=min_weight,
            max_weight=max_weight,
            num_examples=num_examples,
            max_rejection_attempts=max_rejection_attempts,
            max_unique_attempts=max_unique_attempts,
            print_progress_interval=print_progress_interval,
            num_datapoints=num_datapoints or 500000,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )


def generate_curriculum_dataset(
    seed: int = DEFAULT_SEED,
    train_output_file: str = DEFAULT_TRAIN_OUTPUT_FILE,
    val_output_file: str = DEFAULT_VAL_OUTPUT_FILE,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    edge_probability: float = DEFAULT_EDGE_PROBABILITY,
    min_weight: int = DEFAULT_MIN_WEIGHT,
    max_weight: int = DEFAULT_MAX_WEIGHT,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    max_rejection_attempts: int = DEFAULT_MAX_REJECTION_ATTEMPTS,
    max_unique_attempts: int = DEFAULT_MAX_UNIQUE_ATTEMPTS,
    print_progress_interval: int = DEFAULT_PRINT_PROGRESS_INTERVAL,
    rehearsal_ratio: float = DEFAULT_REHEARSAL_RATIO,
    samples_per_stage: int = DEFAULT_SAMPLES_PER_STAGE,
    min_nodes: int = DEFAULT_MIN_NODES,
    max_nodes: int = DEFAULT_MAX_NODES,
):
    """
    Generate dataset using curriculum learning with progressive difficulty.
    Preserves order so easier examples (smaller graphs) come first.
    """
    # Build curriculum stages dynamically
    curriculum_stages = build_curriculum_stages(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        num_examples=num_examples,
        samples_per_stage=samples_per_stage,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    
    all_datapoints = []
    seen_hashes = set()
    previous_stages_datapoints = []  # Track datapoints from previous stages for rehearsal
    
    total_target_samples = sum(stage["samples"] for stage in curriculum_stages)
    
    print(f"Generating curriculum learning dataset with {len(curriculum_stages)} stages...")
    print(f"Graph parameters: Erdős–Rényi G(n, p) with n in [{min_nodes}, {max_nodes}], p = {edge_probability}")
    print(f"Edge weights: scaling with graph size (from 1-{get_scaled_max_weight(min_nodes, max_weight)} at n={min_nodes} to 1-{max_weight} at n={max_nodes})")
    print(f"In-context learning: {num_examples} examples + 1 query per datapoint")
    print(f"Rehearsal ratio: {rehearsal_ratio*100:.1f}% (adding {rehearsal_ratio*100:.1f}% rehearsal samples from previous stages)")
    print(f"Total target samples: {total_target_samples}\n")
    
    # Generate data for each curriculum stage
    for stage_idx, stage in enumerate(curriculum_stages):
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
            datapoint = generate_datapoint(
                num_nodes=num_nodes,
                edge_probability=edge_probability,
                num_examples=num_examples,
                min_weight=stage_min_weight,
                max_weight=stage_max_weight,
                max_rejection_attempts=max_rejection_attempts,
                max_unique_attempts=max_unique_attempts,
            )
            
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
                
                if len(stage_datapoints) % print_progress_interval == 0:
                    print(f"  Generated {len(stage_datapoints)}/{target_samples} samples...")
        
        if len(stage_datapoints) < target_samples:
            print(f"  Warning: Only generated {len(stage_datapoints)}/{target_samples} samples for this stage.")
        
        # Add rehearsal samples from previous stages (if not first stage)
        rehearsal_samples = []
        if stage_idx > 0 and len(previous_stages_datapoints) > 0:
            num_rehearsal = int(target_samples * rehearsal_ratio)
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
    num_val_samples = int(len(all_datapoints) * (1 - train_split))
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
    train_path = Path(__file__).parent / train_output_file
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_dataset, f, separators=(',', ':'))
    
    # Save val dataset
    val_path = Path(__file__).parent / val_output_file
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_dataset, f, separators=(',', ':'))
    
    # Print stage distribution
    print(f"Stage distribution in training set:")
    stage_counts = {}
    for dp in train_datapoints:
        stage = dp.get('curriculum_stage', 'unknown')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    for stage in curriculum_stages:
        stage_name = stage["name"]
        count = stage_counts.get(stage_name, 0)
        percentage = 100 * count / len(train_datapoints) if train_datapoints else 0
        print(f"  {stage_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✓ Train dataset: {train_path} ({len(train_datapoints)} datapoints)")
    print(f"✓ Val dataset: {val_path} ({len(val_datapoints)} datapoints)")
    print(f"✓ Split: {100*train_split:.1f}% train / {100*(1-train_split):.1f}% val")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: Training data is in CURRICULUM ORDER!")
    print("Easy examples (smaller graphs) come first, harder examples (larger graphs) come later.")
    print("Do NOT shuffle the training data!")
    print("\nValidation set is RANDOMLY sampled from the entire dataset.")
    print(f"{'='*80}\n")
    
    return train_datapoints, val_datapoints


def generate_random_dataset(
    seed: int = DEFAULT_SEED,
    train_output_file: str = DEFAULT_TRAIN_OUTPUT_FILE,
    val_output_file: str = DEFAULT_VAL_OUTPUT_FILE,
    train_split: float = DEFAULT_TRAIN_SPLIT,
    edge_probability: float = DEFAULT_EDGE_PROBABILITY,
    min_weight: int = DEFAULT_MIN_WEIGHT,
    max_weight: int = DEFAULT_MAX_WEIGHT,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
    max_rejection_attempts: int = DEFAULT_MAX_REJECTION_ATTEMPTS,
    max_unique_attempts: int = DEFAULT_MAX_UNIQUE_ATTEMPTS,
    print_progress_interval: int = DEFAULT_PRINT_PROGRESS_INTERVAL,
    num_datapoints: int = DEFAULT_NUM_DATAPOINTS,
    min_nodes: int = DEFAULT_MIN_NODES,
    max_nodes: int = DEFAULT_MAX_NODES,
):
    """
    Generate dataset with random node selection (non-curriculum, for backward compatibility).
    """
    num_nodes_list = list(range(min_nodes, max_nodes + 1))
    
    datapoints = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_datapoints * 20  # Prevent infinite loops
    
    print(f"Generating {num_datapoints} unique datapoints (random sampling)...")
    print(f"Graph parameters: Erdős–Rényi G(n, p) with n in [{min_nodes}, {max_nodes}], p = {edge_probability}")
    print(f"Edge weights: integers in [{min_weight}, {max_weight}]")
    print(f"In-context learning: {num_examples} examples + 1 query per datapoint")
    
    while len(datapoints) < num_datapoints and attempts < max_attempts:
        attempts += 1
        
        # Randomly select number of nodes from num_nodes_list
        num_nodes = random.choice(num_nodes_list)
        datapoint = generate_datapoint(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            num_examples=num_examples,
            min_weight=min_weight,
            max_weight=max_weight,
            max_rejection_attempts=max_rejection_attempts,
            max_unique_attempts=max_unique_attempts,
        )
        
        # Skip if generation failed
        if datapoint is None:
            continue
        
        # Check for duplicates
        datapoint_hash = get_datapoint_hash(datapoint)
        if datapoint_hash not in seen_hashes:
            seen_hashes.add(datapoint_hash)
            datapoint['id'] = len(datapoints)
            datapoints.append(datapoint)
            
            if len(datapoints) % print_progress_interval == 0:
                print(f"Generated {len(datapoints)}/{num_datapoints} unique datapoints...")
        else:
            # Duplicate found, skip it
            continue
    
    if len(datapoints) < num_datapoints:
        print(f"Warning: Only generated {len(datapoints)} unique datapoints out of {num_datapoints} requested.")
    
    # Shuffle datapoints before splitting
    random.shuffle(datapoints)
    
    # Split into train and val
    split_idx = int(len(datapoints) * train_split)
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
    train_path = Path(__file__).parent / train_output_file
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_dataset, f, separators=(',', ':'))
    
    # Save val dataset
    val_path = Path(__file__).parent / val_output_file
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_dataset, f, separators=(',', ':'))
    
    print(f"Train dataset saved to {train_path} ({len(train_datapoints)} datapoints)")
    print(f"Val dataset saved to {val_path} ({len(val_datapoints)} datapoints)")
    print(f"Total unique datapoints: {len(datapoints)}")
    print(f"Train/Val split: {len(train_datapoints)}/{len(val_datapoints)} ({100*train_split:.1f}%/{100*(1-train_split):.1f}%)")
    
    return train_datapoints, val_datapoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic shortest path length datasets using curriculum learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset generation parameters
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--train-output-file", type=str, default=DEFAULT_TRAIN_OUTPUT_FILE, help="Path to output training dataset file")
    parser.add_argument("--val-output-file", type=str, default=DEFAULT_VAL_OUTPUT_FILE, help="Path to output validation dataset file")
    parser.add_argument("--train-split", type=float, default=DEFAULT_TRAIN_SPLIT, help="Fraction of data for training (remainder goes to validation)")
    parser.add_argument("--use-curriculum", action="store_true", default=DEFAULT_USE_CURRICULUM, help="Enable curriculum learning")
    parser.add_argument("--no-curriculum", dest="use_curriculum", action="store_false", help="Disable curriculum learning (use random sampling)")
    parser.add_argument("--num-datapoints", type=int, default=None, help="Total number of datapoints (only used if --no-curriculum)")
    
    # Graph generation parameters
    parser.add_argument("--edge-probability", type=float, default=DEFAULT_EDGE_PROBABILITY, help="Probability p for Erdős–Rényi G(n, p) distribution")
    parser.add_argument("--min-weight", type=int, default=DEFAULT_MIN_WEIGHT, help="Minimum edge weight")
    parser.add_argument("--max-weight", type=int, default=DEFAULT_MAX_WEIGHT, help="Maximum edge weight (cap for scaling)")
    
    # In-context learning parameters
    parser.add_argument("--num-examples", type=int, default=DEFAULT_NUM_EXAMPLES, help="Number of example pairs (u_i, v_i) to include")
    parser.add_argument("--max-rejection-attempts", type=int, default=DEFAULT_MAX_REJECTION_ATTEMPTS, help="Maximum attempts for rejection sampling per pair")
    parser.add_argument("--max-unique-attempts", type=int, default=DEFAULT_MAX_UNIQUE_ATTEMPTS, help="Maximum attempts to generate unique examples before regenerating graph")
    
    # Curriculum learning parameters
    parser.add_argument("--samples-per-stage", type=int, default=DEFAULT_SAMPLES_PER_STAGE, help="Number of samples per curriculum stage")
    parser.add_argument("--min-nodes", type=int, default=DEFAULT_MIN_NODES, help="Minimum number of nodes (starting point for curriculum)")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES, help="Maximum number of nodes (ending point for curriculum)")
    parser.add_argument("--rehearsal-ratio", type=float, default=DEFAULT_REHEARSAL_RATIO, help="Ratio of rehearsal samples from previous stages (0.1 = 10%%)")
    
    # Other parameters
    parser.add_argument("--print-progress-interval", type=int, default=DEFAULT_PRINT_PROGRESS_INTERVAL, help="Print progress every N datapoints")
    
    args = parser.parse_args()
    
    generate_dataset(
        seed=args.seed,
        train_output_file=args.train_output_file,
        val_output_file=args.val_output_file,
        train_split=args.train_split,
        use_curriculum=args.use_curriculum,
        edge_probability=args.edge_probability,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        num_examples=args.num_examples,
        max_rejection_attempts=args.max_rejection_attempts,
        max_unique_attempts=args.max_unique_attempts,
        print_progress_interval=args.print_progress_interval,
        rehearsal_ratio=args.rehearsal_ratio,
        samples_per_stage=args.samples_per_stage,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        num_datapoints=args.num_datapoints,
    )
