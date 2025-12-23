#!/usr/bin/env python3
"""
Generate dataset for TRUE ICL with SET TARGET evaluation (0, 2, 4-shot).

TRAINING:
- Single examples only (no few-shot examples in prompts)
- Train on sizes: [3, 4, 6, 8, 9, 10]
- Each example is just: instruction → output
- Format: "Find shortest path from node X to node Y"

EVALUATION:
- Test if model can learn NEW task format from examples
- Eval on sizes: [5, 7] (unknown/unseen)
- NEW TASK: "Find shortest path from node X to any node in set S"
- Few-shot: Construct prompts with 0, 2, or 4 examples + 1 query at inference time
"""

import json
import random
import numpy as np
from typing import List, Tuple, Optional, Set
import heapq
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Node sizes for train and validation
TRAIN_NODE_SIZES = [3, 4, 6, 8, 9, 10]
VAL_NODE_SIZES_UNKNOWN = [5, 7]  # Sizes not seen during training (unknown) - ONLY these used for eval

# Number of samples per node size
SAMPLES_PER_SIZE = 6667  # Same as shuffled dataset

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_icl_true.json"  # Same training data
VAL_OUTPUT_FILE = "data/val_icl_true_set_target_024shot.json"  # Different validation with set targets

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

PRINT_PROGRESS_INTERVAL = 1000
USE_CHAIN_OF_THOUGHT = True

# For validation: set target parameters
# NO REUSE: 50 queries per size × 2 sizes = 100 queries total
#           100 queries × 0 examples (zero-shot) = 0 examples
#           100 queries × 2 examples (2-shot) = 200 examples
#           100 queries × 4 examples (4-shot) = 400 examples
#           Total: 100 queries + 600 examples = 700 unique graphs
VAL_SAMPLES_TOTAL_UNKNOWN = 700  # 700 samples total for unknown sizes (350 per size)


def dijkstra_all_reachable(adjacency_matrix: np.ndarray, start: int) -> List[int]:
    """Find all nodes reachable from start using Dijkstra's algorithm."""
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    visited = [False] * n
    
    pq = [(0, start)]
    reachable = []
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        reachable.append(current)
        
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return reachable


def dijkstra_to_set_with_trace(adjacency_matrix: np.ndarray, start: int, target_set: Set[int]) -> Tuple[Optional[List[int]], int, int, List[str]]:
    """
    Compute shortest path from start to ANY node in target_set using Dijkstra's algorithm.
    
    Returns:
        (path, total_distance, reached_target, reasoning_steps)
        - path: shortest path to closest node in target_set
        - total_distance: distance to that node
        - reached_target: which node in target_set was reached
        - reasoning_steps: chain-of-thought reasoning
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    reasoning = []
    
    reasoning.append(f"Starting at node {start} with distance 0")
    reasoning.append(f"Target set: {sorted(target_set)}")
    
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        reasoning.append(f"Visiting node {current} (distance: {current_dist})")
        
        # Check if we reached any target node
        if current in target_set:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = previous[node]
            path = path[::-1]
            
            reasoning.append(f"Reached target node {current} (in target set {sorted(target_set)})")
            reasoning.append(f"Reconstructed path: {path}")
            
            return path, int(current_dist), current, reasoning
        
        # Check all neighbors
        neighbors_explored = []
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    old_dist = distances[neighbor]
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    neighbors_explored.append(f"node {neighbor} (dist: {new_dist})")
        
        if neighbors_explored:
            reasoning.append(f"Updated distances for: {', '.join(neighbors_explored)}")
    
    # No path found to any target node
    reasoning.append(f"No path found to any node in target set {sorted(target_set)}")
    return None, -1, -1, reasoning


def generate_random_graph(num_nodes: int) -> np.ndarray:
    """Generate a random directed graph with weighted edges."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < EDGE_PROBABILITY:
                weight = random.randint(MIN_WEIGHT, MAX_WEIGHT)
                adjacency[i][j] = weight
    
    return adjacency


def generate_datapoint(num_nodes: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate a single datapoint with set target format."""
    max_attempts = 20 * num_nodes  # More attempts for larger graphs
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        # Check if graph is unique (by adjacency matrix)
        graph_hash = hashlib.md5(str(adjacency.tolist()).encode()).hexdigest()
        if graph_hash in seen_graph_hashes:
            continue
        
        # Find all reachable nodes from each starting point
        all_reachable = {}
        for start in range(num_nodes):
            reachable = dijkstra_all_reachable(adjacency, start)
            all_reachable[start] = reachable
        
        # Pick a random start node that has at least one reachable node
        valid_starts = [s for s in range(num_nodes) if len(all_reachable[s]) > 1]
        if not valid_starts:
            continue
        
        start_node = random.choice(valid_starts)
        
        # Pick a target set (size 1 to n-1, excluding start_node)
        reachable_from_start = [n for n in all_reachable[start_node] if n != start_node]
        if not reachable_from_start:
            continue
        
        # Target set size: 1 to min(len(reachable_from_start), num_nodes - 1)
        max_set_size = min(len(reachable_from_start), num_nodes - 1)
        set_size = random.randint(1, max_set_size)
        target_set = set(random.sample(reachable_from_start, set_size))
        
        # Find shortest path to any node in target set
        path, distance, reached_target, reasoning_steps = dijkstra_to_set_with_trace(
            adjacency, start_node, target_set
        )
        
        if path is None or distance < 0:
            continue
        
        # Mark graph as used
        seen_graph_hashes.add(graph_hash)
        
        # Build output text
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(reasoning_steps)
            output_text = (
                f"I'll find the shortest path from node {start_node} to any node in the set {sorted(target_set)}:\n\n"
                f"{reasoning_text}\n\n"
                f"Final Answer: {json.dumps(path)} with a total distance of {distance}."
            )
        else:
            output_text = f"Final Answer: {json.dumps(path)} with a total distance of {distance}."
        
        datapoint = {
            "adjacency_matrix": adjacency.tolist(),
            "num_nodes": num_nodes,
            "start_node": start_node,
            "target_set": sorted(list(target_set)),  # Store as sorted list for consistency
            "reached_target": reached_target,  # Which node in target set was reached
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": reasoning_steps if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        return datapoint
    
    return None


def build_single_example_prompt(datapoint: dict) -> dict:
    """Build a single-example prompt format (adds instruction/input/output fields for Unsloth compatibility).
    
    Note: For validation data, this adds the fields but the callback will construct ICL prompts
    from the raw fields (adjacency_matrix, start_node, target_set) at inference time.
    """
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    
    # For validation: use the set-target format (what we're evaluating)
    # But we still need instruction/input fields for Unsloth's formatting function
    target_set_str = ", ".join(map(str, datapoint["target_set"]))
    instruction = (
        f"Graph representation: {adjacency}\n"
        f"Find the shortest path from node {datapoint['start_node']} to any node in the set {{{target_set_str}}}.\n"
        "Nodes are indexed from 0.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' "
        "where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    return {
        "instruction": instruction,
        "input": "",
        "output": datapoint.get("output", ""),  # Already created in generate_datapoint
        # Preserve all raw fields for callback
        "adjacency_matrix": datapoint["adjacency_matrix"],
        "num_nodes": datapoint["num_nodes"],
        "start_node": datapoint["start_node"],
        "target_set": datapoint["target_set"],
        "reached_target": datapoint["reached_target"],
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"\n{'='*80}")
    print("TRUE ICL SET TARGET DATA GENERATION (0, 2, 4-shot)")
    print(f"{'='*80}\n")
    print(f"Training: Single examples only (no few-shot)")
    print(f"Evaluation: Will test NEW task format - shortest path to any node in a set")
    print(f"  - Training format: 'Find path from X to Y'")
    print(f"  - Eval format: 'Find path from X to any node in set S'")
    print(f"  - Model must learn this new format from examples at inference time\n")
    
    # Track unique graphs across all sizes
    seen_graph_hashes = set()
    
    # Generate unknown validation samples (ONLY these for eval)
    print(f"\n--- UNKNOWN VALIDATION SAMPLES (sizes not seen during training) ---")
    print(f"   Generating {VAL_SAMPLES_TOTAL_UNKNOWN} samples total (distributed across sizes {VAL_NODE_SIZES_UNKNOWN})")
    print(f"   Ensuring all graphs are unique (by adjacency matrix)")
    print(f"   Task: Find shortest path from start node to ANY node in target set")
    print(f"   Target set size: 1 to n-1 (where n = number of nodes)")
    print(f"   Evaluation: 0-shot, 2-shot, and 4-shot")
    print(f"   Breakdown: 100 queries + 200 examples (2-shot) + 400 examples (4-shot) = 700 total\n")
    
    val_samples = []
    samples_per_size = VAL_SAMPLES_TOTAL_UNKNOWN // len(VAL_NODE_SIZES_UNKNOWN)
    
    for size in VAL_NODE_SIZES_UNKNOWN:
        print(f"Generating {samples_per_size} samples for size {size}...")
        size_samples = []
        size_seen_hashes = set()
        
        for i in range(samples_per_size):
            if (i + 1) % PRINT_PROGRESS_INTERVAL == 0:
                print(f"  Progress: {i + 1}/{samples_per_size}")
            
            # Use combined seen hashes (across all sizes and training)
            datapoint = generate_datapoint(size, seen_graph_hashes)
            
            if datapoint is None:
                print(f"  Warning: Failed to generate unique datapoint {i + 1} for size {size}")
                continue
            
            size_samples.append(datapoint)
            size_seen_hashes.add(hashlib.md5(str(datapoint["adjacency_matrix"]).encode()).hexdigest())
        
        print(f"  Generated {len(size_samples)} unique samples for size {size}")
        val_samples.extend(size_samples)
    
    print(f"\n✅ Generated {len(val_samples)} total validation samples")
    print(f"   Unique graphs: {len(seen_graph_hashes)}")
    
    # Convert to instruction/input/output format (for Unsloth compatibility)
    # while preserving raw fields for callback
    print(f"\nConverting to instruction/input/output format...")
    val_dataset = [build_single_example_prompt(dp) for dp in val_samples]
    
    # Save validation dataset
    val_output_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output_path, 'w') as f:
        json.dump(val_dataset, f, indent=2)
    print(f"✅ Saved validation dataset to {val_output_path}")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: This is TRUE ICL setup with SET TARGET evaluation (0, 2, 4-shot)")
    print("  - Training: Single examples only (query only, no few-shot examples)")
    print("  - Training format: 'Find shortest path from node X to node Y'")
    print("  - Evaluation: Only unknown sizes [5, 7] - 50 queries per size = 100 queries total")
    print("  - Evaluation format: 'Find shortest path from node X to any node in set S'")
    print("  - Evaluation modes: 0-shot, 2-shot, and 4-shot")
    print("  - Total completions: 50 per size × 2 sizes × 3 modes = 300 completions")
    print("  - Model must learn the NEW task format from examples at inference time")
    print("  - Model learns from context only during evaluation, not during training")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

