#!/usr/bin/env python3
"""
Generate dataset for TRUE ICL with SAME PARITY evaluation (0, 2, 4-shot).
ONLY SIZE 6 - Training and evaluation both on size 6.

TRAINING:
- Single examples only (no few-shot examples in prompts)
- Train on size: 6 ONLY
- Each example is just: instruction → output
- Format: "Find shortest path from node X to node Y"

EVALUATION:
- Test if model can learn NEW task format from examples
- Eval on size: 6 ONLY
- NEW TASK: "Find shortest path from node X to node Y only using nodes of the same parity"
- Few-shot: Construct prompts with 0, 2, or 4 examples + 1 query at inference time
"""

import json
import random
import numpy as np
from typing import List, Tuple, Optional
import heapq
import hashlib
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Node sizes for train and validation
TRAIN_NODE_SIZE = 6  # Only size 6 for training
VAL_NODE_SIZE = 6  # Only size 6 for evaluation

# Number of samples
TRAIN_SAMPLES = 10000  # Training samples for size 6
VAL_SAMPLES_TOTAL = 700  # 700 samples total for validation (same as other scripts)

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_icl_true_size6_only.json"  # Training data only for size 6
VAL_OUTPUT_FILE = "data/val_icl_true_same_parity_024shot_size6_only.json"  # Validation with same parity constraint for size 6

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

PRINT_PROGRESS_INTERVAL = 1000
USE_CHAIN_OF_THOUGHT = True

# For validation: same parity parameters
# NO REUSE: 50 queries × 0 examples (zero-shot) = 0 examples
#           50 queries × 2 examples (2-shot) = 100 examples
#           50 queries × 4 examples (4-shot) = 200 examples
#           Total: 50 queries + 300 examples = 350 unique graphs
# But we'll generate 700 total to have enough for the example pool
VAL_QUERIES = 50  # 50 queries for evaluation


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


def dijkstra_with_trace(adjacency_matrix: np.ndarray, start: int, end: int) -> Tuple[Optional[List[int]], int, List[str]]:
    """
    Compute shortest path with reasoning trace for chain-of-thought (standard, no constraints).
    
    Returns:
        (path, total_distance, reasoning_steps)
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    reasoning = []
    
    reasoning.append(f"Starting at node {start} with distance 0")
    
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        reasoning.append(f"Visiting node {current} (distance: {current_dist})")
        
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = previous[node]
            path = path[::-1]
            
            reasoning.append(f"Reached destination node {end}")
            reasoning.append(f"Reconstructed path: {path}")
            
            return path, int(current_dist), reasoning
        
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
            reasoning.append(f"  Updated distances for: {', '.join(neighbors_explored)}")
    
    return None, -1, reasoning + ["No path found"]


def dijkstra_same_parity_with_trace(adjacency_matrix: np.ndarray, start: int, end: int) -> Tuple[Optional[List[int]], int, List[str]]:
    """
    Compute shortest path from start to end only using nodes of the same parity as start.
    
    Returns:
        (path, total_distance, reasoning_steps)
        - path: shortest path that only uses nodes with same parity as start
        - total_distance: distance of that path
        - reasoning_steps: chain-of-thought reasoning
    """
    n = len(adjacency_matrix)
    start_parity = start % 2  # 0 for even, 1 for odd
    
    # Check if end has same parity as start
    if end % 2 != start_parity:
        return None, -1, [f"Start node {start} and end node {end} have different parity. No path exists."]
    
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    reasoning = []
    
    reasoning.append(f"Starting at node {start} (parity: {'even' if start_parity == 0 else 'odd'}) with distance 0")
    reasoning.append(f"Target node: {end} (must have same parity)")
    reasoning.append(f"Can only visit nodes with {'even' if start_parity == 0 else 'odd'} parity")
    
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if visited[current]:
            continue
        
        visited[current] = True
        reasoning.append(f"Visiting node {current} (distance: {current_dist}, parity: {'even' if current % 2 == 0 else 'odd'})")
        
        if current == end:
            # Reconstruct path
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = previous[node]
            path = path[::-1]
            
            reasoning.append(f"Reached destination node {end}")
            reasoning.append(f"Reconstructed path: {path}")
            
            return path, int(current_dist), reasoning
        
        # Check all neighbors (but only visit nodes with same parity)
        neighbors_explored = []
        for neighbor in range(n):
            # Only consider neighbors with same parity as start
            if neighbor % 2 != start_parity:
                continue
                
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    old_dist = distances[neighbor]
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    neighbors_explored.append(f"node {neighbor} (dist: {new_dist}, edge: {edge_weight})")
        
        if neighbors_explored:
            reasoning.append(f"Updated distances for: {', '.join(neighbors_explored)}")
    
    # No path found
    reasoning.append(f"No path found from {start} to {end} using only {'even' if start_parity == 0 else 'odd'} nodes")
    return None, -1, reasoning


def generate_random_graph(num_nodes: int) -> np.ndarray:
    """Generate a random directed graph with weighted edges."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < EDGE_PROBABILITY:
                weight = random.randint(MIN_WEIGHT, MAX_WEIGHT)
                adjacency[i][j] = weight
    
    return adjacency


def generate_training_datapoint(num_nodes: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate a single training datapoint (standard shortest path, no constraints)."""
    max_attempts = 20 * num_nodes
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        # Check if graph is unique
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
        
        # Pick an end node (must be reachable from start, different from start)
        reachable_from_start = [n for n in all_reachable[start_node] if n != start_node]
        if not reachable_from_start:
            continue
        
        end_node = random.choice(reachable_from_start)
        
        # Find shortest path (standard, no constraints)
        path, distance, reasoning_steps = dijkstra_with_trace(
            adjacency, start_node, end_node
        )
        
        if path is None or distance < 0:
            continue
        
        # Mark graph as used
        seen_graph_hashes.add(graph_hash)
        
        # Build output text
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(reasoning_steps)
            output_text = (
                f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
                f"{reasoning_text}\n\n"
                f"Final Answer: The shortest path from node {start_node} "
                f"to node {end_node} is {json.dumps(path)} "
                f"with total distance {distance}."
            )
        else:
            output_text = f"Final Answer: {json.dumps(path)} with a total distance of {distance}."
        
        datapoint = {
            "adjacency_matrix": adjacency.tolist(),
            "num_nodes": num_nodes,
            "start_node": start_node,
            "end_node": end_node,
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": reasoning_steps if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        return datapoint
    
    return None


def generate_validation_datapoint(num_nodes: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate a single validation datapoint with same parity format."""
    max_attempts = 20 * num_nodes
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        # Check if graph is unique
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
        start_parity = start_node % 2
        
        # Pick an end node (must be reachable from start, different from start, and same parity)
        reachable_from_start = [n for n in all_reachable[start_node] if n != start_node and n % 2 == start_parity]
        if not reachable_from_start:
            continue
        
        end_node = random.choice(reachable_from_start)
        
        # Find shortest path with same parity constraint
        path, distance, reasoning_steps = dijkstra_same_parity_with_trace(
            adjacency, start_node, end_node
        )
        
        if path is None or distance < 0:
            continue
        
        # Mark graph as used
        seen_graph_hashes.add(graph_hash)
        
        # Build output text
        parity_str = "even" if start_parity == 0 else "odd"
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(reasoning_steps)
            output_text = (
                f"I'll find the shortest path from node {start_node} to node {end_node} "
                f"only using nodes with {parity_str} parity:\n\n"
                f"{reasoning_text}\n\n"
                f"Final Answer: {json.dumps(path)} with a total distance of {distance}."
            )
        else:
            output_text = f"Final Answer: {json.dumps(path)} with a total distance of {distance}."
        
        datapoint = {
            "adjacency_matrix": adjacency.tolist(),
            "num_nodes": num_nodes,
            "start_node": start_node,
            "end_node": end_node,
            "parity": parity_str,
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": reasoning_steps if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        return datapoint
    
    return None


def build_training_prompt(datapoint: dict) -> dict:
    """Build training prompt format (standard shortest path)."""
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    
    input_text = (
        f"Graph representation: {adjacency}\n"
        f"This is a {datapoint['num_nodes']}x{datapoint['num_nodes']} adjacency matrix "
        f"where 0 means no edge and positive numbers represent edge weights. "
        f"Nodes are indexed from 0 to {datapoint['num_nodes']-1}."
    )
    
    instruction = (
        f"In this directed graph, find the shortest path from node {datapoint['start_node']} "
        f"to node {datapoint['end_node']} using Dijkstra's algorithm. "
        f"Show your reasoning step by step."
    )
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": datapoint.get("output", ""),
        # Preserve raw fields
        "adjacency_matrix": datapoint["adjacency_matrix"],
        "num_nodes": datapoint["num_nodes"],
        "start_node": datapoint["start_node"],
        "end_node": datapoint["end_node"],
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def build_validation_prompt(datapoint: dict) -> dict:
    """Build validation prompt format (same parity constraint)."""
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    parity_str = datapoint.get("parity", "even" if datapoint["start_node"] % 2 == 0 else "odd")
    
    instruction = (
        f"Graph representation: {adjacency}\n"
        f"Find the shortest path from node {datapoint['start_node']} to node {datapoint['end_node']} "
        f"only using nodes with {parity_str} parity.\n"
        "Nodes are indexed from 0.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' "
        "where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    return {
        "instruction": instruction,
        "input": "",
        "output": datapoint.get("output", ""),
        # Preserve all raw fields for callback
        "adjacency_matrix": datapoint["adjacency_matrix"],
        "num_nodes": datapoint["num_nodes"],
        "start_node": datapoint["start_node"],
        "end_node": datapoint["end_node"],
        "parity": datapoint.get("parity", "even" if datapoint["start_node"] % 2 == 0 else "odd"),
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"\n{'='*80}")
    print("TRUE ICL SAME PARITY DATA GENERATION (0, 2, 4-shot) - SIZE 6 ONLY")
    print(f"{'='*80}\n")
    print(f"Training: Single examples only (no few-shot)")
    print(f"Training size: {TRAIN_NODE_SIZE} ONLY")
    print(f"Evaluation: Will test NEW task format - shortest path with same parity constraint")
    print(f"Evaluation size: {VAL_NODE_SIZE} ONLY")
    print(f"  - Training format: 'Find path from X to Y'")
    print(f"  - Eval format: 'Find path from X to Y only using nodes of the same parity'")
    print(f"  - Model must learn this new format from examples at inference time\n")
    
    # Track unique graphs
    seen_graph_hashes = set()
    
    # Generate training samples (standard shortest path, size 6 only)
    print(f"\n--- TRAINING SAMPLES (size {TRAIN_NODE_SIZE} only) ---")
    print(f"   Generating {TRAIN_SAMPLES} training samples")
    print(f"   Task: Standard shortest path (no constraints)")
    print(f"   Format: 'Find shortest path from node X to node Y'\n")
    
    train_samples = []
    for i in range(TRAIN_SAMPLES):
        if (i + 1) % PRINT_PROGRESS_INTERVAL == 0:
            print(f"  Progress: {i + 1}/{TRAIN_SAMPLES}")
        
        datapoint = generate_training_datapoint(TRAIN_NODE_SIZE, seen_graph_hashes)
        
        if datapoint is None:
            print(f"  Warning: Failed to generate unique datapoint {i + 1}")
            continue
        
        train_samples.append(datapoint)
    
    print(f"✅ Generated {len(train_samples)} training samples")
    
    # Generate validation samples (same parity constraint, size 6 only)
    print(f"\n--- VALIDATION SAMPLES (size {VAL_NODE_SIZE} only) ---")
    print(f"   Generating {VAL_SAMPLES_TOTAL} validation samples")
    print(f"   Task: Find shortest path from start to end only using nodes with same parity")
    print(f"   Constraint: Start and end must have same parity (both even or both odd)")
    print(f"   Evaluation: 0-shot, 2-shot, and 4-shot")
    print(f"   Breakdown: {VAL_QUERIES} queries + 100 examples (2-shot) + 200 examples (4-shot) = {VAL_SAMPLES_TOTAL} total\n")
    
    val_samples = []
    for i in range(VAL_SAMPLES_TOTAL):
        if (i + 1) % PRINT_PROGRESS_INTERVAL == 0:
            print(f"  Progress: {i + 1}/{VAL_SAMPLES_TOTAL}")
        
        datapoint = generate_validation_datapoint(VAL_NODE_SIZE, seen_graph_hashes)
        
        if datapoint is None:
            print(f"  Warning: Failed to generate unique datapoint {i + 1}")
            continue
        
        val_samples.append(datapoint)
    
    print(f"✅ Generated {len(val_samples)} validation samples")
    print(f"   Unique graphs: {len(seen_graph_hashes)}")
    
    # Convert to instruction/input/output format
    print(f"\nConverting to instruction/input/output format...")
    train_dataset = [build_training_prompt(dp) for dp in train_samples]
    val_dataset = [build_validation_prompt(dp) for dp in val_samples]
    
    # Save datasets
    train_output_path = Path(__file__).parent / TRAIN_OUTPUT_FILE
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output_path, 'w') as f:
        json.dump(train_dataset, f, indent=2)
    print(f"✅ Saved training dataset to {train_output_path}")
    
    val_output_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output_path, 'w') as f:
        json.dump(val_dataset, f, indent=2)
    print(f"✅ Saved validation dataset to {val_output_path}")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: This is TRUE ICL setup with SAME PARITY evaluation (0, 2, 4-shot) - SIZE 6 ONLY")
    print("  - Training: Single examples only (query only, no few-shot examples)")
    print(f"  - Training: Size {TRAIN_NODE_SIZE} only - {len(train_samples)} samples")
    print("  - Training format: 'Find shortest path from node X to node Y'")
    print(f"  - Evaluation: Size {VAL_NODE_SIZE} only - {VAL_QUERIES} queries")
    print("  - Evaluation format: 'Find shortest path from node X to node Y only using nodes of the same parity'")
    print("  - Evaluation modes: 0-shot, 2-shot, and 4-shot")
    print(f"  - Total completions: {VAL_QUERIES} queries × 3 modes = {VAL_QUERIES * 3} completions")
    print("  - Model must learn the NEW task format from examples at inference time")
    print("  - Model learns from context only during evaluation, not during training")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

