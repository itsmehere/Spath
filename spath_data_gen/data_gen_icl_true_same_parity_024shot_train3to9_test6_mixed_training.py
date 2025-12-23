#!/usr/bin/env python3
"""
Generate dataset for TRUE ICL with SAME PARITY evaluation (0, 2, 4-shot).
TRAIN ON SIZES 3-9, TEST ON SIZE 6.

TRAINING:
- Mixed training: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot (equal proportions)
- Train on sizes: 3, 4, 5, 6, 7, 8, 9
- Format matches inference format (not alpaca)
- Shuffled graph sizes and shuffled prompts with/without examples together

EVALUATION:
- Test if model can learn NEW task format from examples
- Eval on size: 6 ONLY
- NEW TASK: "Find shortest path from node X to node Y only using nodes of the same parity"
- Few-shot: Construct prompts with 0, 2, or 4 examples + 1 query at inference time
- Constraint: Path must only use nodes with same parity (even/odd) as start node
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
TRAIN_NODE_SIZES = [3, 4, 5, 6, 7, 8, 9]  # Train on sizes 3 through 9
VAL_NODE_SIZE = 6  # Test on size 6 only

# Number of samples
TRAIN_SAMPLES_TOTAL = 10000  # Total training samples across all sizes
VAL_SAMPLES_TOTAL = 700  # 700 samples total for validation

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_icl_true_same_parity_train3to9_test6_mixed_training.json"
VAL_OUTPUT_FILE = "data/val_icl_true_same_parity_024shot_train3to9_test6_mixed_training.json"

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

PRINT_PROGRESS_INTERVAL = 1000
USE_CHAIN_OF_THOUGHT = True

# For validation: same parity parameters
VAL_QUERIES = 50  # 50 queries for evaluation

# Training: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot (equal proportions)
TRAIN_ZERO_SHOT_RATIO = 1/3
TRAIN_TWO_SHOT_RATIO = 1/3
TRAIN_FOUR_SHOT_RATIO = 1/3


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


def dijkstra_standard(adjacency_matrix: np.ndarray, start: int, end: int) -> Tuple[Optional[List[int]], int]:
    """Compute standard shortest path without any constraints.
    
    Returns:
        (path, total_distance)
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    
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
            path = path[::-1]
            return path, int(current_dist)
        
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return None, -1


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
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                # Only consider neighbors with same parity as start
                if neighbor % 2 != start_parity:
                    continue
                
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
        
        # Build output text (inference format)
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(reasoning_steps)
            output_text = (
                f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
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
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": reasoning_steps if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        return datapoint
    
    return None


def generate_validation_datapoint(num_nodes: int, seen_graph_hashes: set, allow_used_graphs: bool = False) -> Optional[dict]:
    """Generate a single validation datapoint with same parity format."""
    max_attempts = 50 * num_nodes
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        # Check if graph is unique
        graph_hash = hashlib.md5(str(adjacency.tolist()).encode()).hexdigest()
        if not allow_used_graphs and graph_hash in seen_graph_hashes:
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
        start_parity = start_node % 2  # 0 for even, 1 for odd
        
        # Pick an end node that has the same parity as start and is reachable
        reachable_from_start = [n for n in all_reachable[start_node] 
                               if n != start_node and n % 2 == start_parity]
        if not reachable_from_start:
            continue
        
        end_node = random.choice(reachable_from_start)
        
        # Find shortest path using same parity constraint
        path, distance, reasoning_steps = dijkstra_same_parity_with_trace(
            adjacency, start_node, end_node
        )
        
        if path is None or distance < 0:
            continue
        
        # Mark graph as used
        if not allow_used_graphs:
            seen_graph_hashes.add(graph_hash)
        
        # Build output text (inference format)
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(reasoning_steps)
            output_text = (
                f"I'll find the shortest path from node {start_node} to node {end_node} only using nodes of the same parity:\n\n"
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
            "same_parity": True,  # Mark as same parity format
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": reasoning_steps if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text,
        }
        
        return datapoint
    
    return None


def build_zero_shot_training_prompt(datapoint: dict) -> dict:
    """Build zero-shot training prompt in inference format (not alpaca)."""
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    
    # Use inference format (matches _build_zero_shot_prompt)
    text = (
        f"Graph representation: {adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {datapoint['start_node']} to node {datapoint['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' "
        "where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    return {
        "text": text + "\n" + datapoint.get("output", ""),  # Prompt + response in inference format
        # Preserve raw fields
        "adjacency_matrix": datapoint["adjacency_matrix"],
        "num_nodes": datapoint["num_nodes"],
        "start_node": datapoint["start_node"],
        "end_node": datapoint["end_node"],
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def build_few_shot_training_prompt(query_datapoint: dict, example_datapoints: List[dict]) -> dict:
    """Build few-shot training prompt in inference format (not alpaca).
    
    Args:
        query_datapoint: The query datapoint
        example_datapoints: List of example datapoints (can be 2 or 4 for mixed training)
    """
    parts = []
    
    # Add header (matches _build_icl_prompt)
    parts.append(f"Below are {len(example_datapoints)} examples. Later, you will be asked to solve a similar problem.\n")
    
    # Add examples with answers (matches _build_icl_prompt format)
    for idx, example in enumerate(example_datapoints):
        adjacency = json.dumps(example["adjacency_matrix"])
        parts.append(
            f"Example {idx + 1}:\n"
            f"Graph representation: {adjacency}\n"
            f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
            f"Find the shortest path from node {example['start_node']} to node {example['end_node']}.\n"
            "Nodes are indexed from 0.\n"
        )
        
        if example.get('reasoning_steps'):
            reasoning_text = "\n".join(example['reasoning_steps'])
            parts.append(
                f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
                f"{reasoning_text}\n\n"
            )
        
        parts.append(
            f"Final Answer: {json.dumps(example.get('shortest_path', []))} with a total distance of {example.get('total_distance', 'unknown')}.\n"
        )
    
    # Add separator
    parts.append("\nNow, here is a question for you to solve:\n")
    
    # Add query (matches _build_icl_prompt format)
    query_adjacency = json.dumps(query_datapoint["adjacency_matrix"])
    parts.append(
        f"Question:\n"
        f"Graph representation: {query_adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {query_datapoint['start_node']} to node {query_datapoint['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "Use Dijkstra's algorithm to compute the shortest path for the question graph.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    prompt_text = "\n".join(parts)
    
    return {
        "text": prompt_text + "\n" + query_datapoint.get("output", ""),  # Prompt + response in inference format
        # Preserve raw fields
        "adjacency_matrix": query_datapoint["adjacency_matrix"],
        "num_nodes": query_datapoint["num_nodes"],
        "start_node": query_datapoint["start_node"],
        "end_node": query_datapoint["end_node"],
        "shortest_path": query_datapoint["shortest_path"],
        "total_distance": query_datapoint["total_distance"],
        "reasoning_steps": query_datapoint.get("reasoning_steps", []),
    }


def build_validation_prompt(datapoint: dict) -> dict:
    """Build validation prompt format (same parity constraint) - for callback use."""
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    
    instruction = (
        f"Graph representation: {adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {datapoint['start_node']} to node {datapoint['end_node']} only using nodes of the same parity.\n"
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
        "same_parity": datapoint.get("same_parity", True),
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"\n{'='*80}")
    print("TRUE ICL SAME PARITY DATA GENERATION (0, 2, 4-shot)")
    print("TRAIN ON SIZES 3-9, TEST ON SIZE 6")
    print("TRAINING: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot (equal proportions)")
    print("TRAINING: Uses inference format (not alpaca)")
    print(f"{'='*80}\n")
    print(f"Training: Mixed format (1/3 zero-shot, 1/3 2-shot, 1/3 4-shot)")
    print(f"Training sizes: {TRAIN_NODE_SIZES}")
    print(f"Evaluation: Will test NEW task format - shortest path using same parity nodes")
    print(f"Evaluation size: {VAL_NODE_SIZE} ONLY")
    print(f"  - Training format: 'Find path from X to Y' (matches inference format)")
    print(f"  - Eval format: 'Find path from X to Y only using nodes of the same parity'")
    print(f"  - Constraint: Path must only use nodes with same parity (even/odd) as start node")
    print(f"  - Model must learn this new format from examples at inference time\n")
    
    # Track unique graphs
    seen_graph_hashes = set()
    
    # Generate training samples (standard shortest path, sizes 3-9)
    print(f"\n--- TRAINING SAMPLES (sizes {TRAIN_NODE_SIZES}) ---")
    print(f"   Generating {TRAIN_SAMPLES_TOTAL} training samples across {len(TRAIN_NODE_SIZES)} sizes")
    print(f"   Task: Standard shortest path (no constraints)")
    print(f"   Format: 'Find shortest path from node X to node Y' (inference format)")
    print(f"   Distribution: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot\n")
    
    train_datapoints = []
    samples_per_size = TRAIN_SAMPLES_TOTAL // len(TRAIN_NODE_SIZES)
    remaining_samples = TRAIN_SAMPLES_TOTAL % len(TRAIN_NODE_SIZES)
    
    for size_idx, node_size in enumerate(TRAIN_NODE_SIZES):
        size_samples = samples_per_size + (1 if size_idx < remaining_samples else 0)
        print(f"  Generating {size_samples} samples for size {node_size}...")
        
        for i in range(size_samples):
            if (i + 1) % PRINT_PROGRESS_INTERVAL == 0:
                print(f"    Progress: {i + 1}/{size_samples}")
            
            datapoint = generate_training_datapoint(node_size, seen_graph_hashes)
            
            if datapoint is None:
                print(f"    Warning: Failed to generate unique datapoint {i + 1} for size {node_size}")
                continue
            
            train_datapoints.append(datapoint)
        
        print(f"  ✅ Generated {size_samples} samples for size {node_size}")
    
    print(f"\n✅ Generated {len(train_datapoints)} total training datapoints")
    
    # Now build training prompts: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot
    print(f"\n--- BUILDING TRAINING PROMPTS (1/3 zero-shot, 1/3 2-shot, 1/3 4-shot) ---")
    num_zero_shot = int(len(train_datapoints) * TRAIN_ZERO_SHOT_RATIO)
    num_two_shot = int(len(train_datapoints) * TRAIN_TWO_SHOT_RATIO)
    num_four_shot = len(train_datapoints) - num_zero_shot - num_two_shot  # Remaining for 4-shot
    
    print(f"   Zero-shot prompts: {num_zero_shot}")
    print(f"   2-shot prompts: {num_two_shot}")
    print(f"   4-shot prompts: {num_four_shot}")
    print(f"   Total: {len(train_datapoints)}\n")
    
    # Shuffle datapoints to mix sizes
    random.shuffle(train_datapoints)
    
    # Split into zero-shot, 2-shot, and 4-shot groups
    zero_shot_datapoints = train_datapoints[:num_zero_shot]
    two_shot_query_datapoints = train_datapoints[num_zero_shot:num_zero_shot + num_two_shot]
    four_shot_query_datapoints = train_datapoints[num_zero_shot + num_two_shot:]
    
    # Build zero-shot prompts
    train_dataset = []
    for dp in zero_shot_datapoints:
        train_dataset.append(build_zero_shot_training_prompt(dp))
    
    # Build 2-shot and 4-shot prompts (need to pair queries with examples)
    example_pool = train_datapoints.copy()  # All datapoints can be examples
    
    def select_distinct_examples(query_dp, example_pool, num_needed):
        """Select distinct examples for a query."""
        candidate_examples = [ex for ex in example_pool 
                            if ex is not query_dp 
                            and ex["adjacency_matrix"] != query_dp["adjacency_matrix"]]
        
        if len(candidate_examples) < num_needed:
            # Fallback: use any examples if not enough distinct ones
            candidate_examples = [ex for ex in example_pool if ex is not query_dp]
        
        if len(candidate_examples) >= num_needed:
            selected_examples = random.sample(candidate_examples, min(num_needed * 2, len(candidate_examples)))
            # Ensure examples are distinct from each other
            distinct_examples = []
            seen_graphs = set()
            for ex in selected_examples:
                ex_hash = hashlib.md5(str(ex["adjacency_matrix"]).encode()).hexdigest()
                if ex_hash not in seen_graphs:
                    distinct_examples.append(ex)
                    seen_graphs.add(ex_hash)
                if len(distinct_examples) >= num_needed:
                    break
            
            if len(distinct_examples) >= num_needed:
                return distinct_examples[:num_needed]
        
        return None
    
    # Build 2-shot prompts
    for query_dp in two_shot_query_datapoints:
        distinct_examples = select_distinct_examples(query_dp, example_pool, 2)
        if distinct_examples:
            train_dataset.append(build_few_shot_training_prompt(query_dp, distinct_examples))
        else:
            # Fallback to zero-shot if can't find enough distinct examples
            train_dataset.append(build_zero_shot_training_prompt(query_dp))
    
    # Build 4-shot prompts
    for query_dp in four_shot_query_datapoints:
        distinct_examples = select_distinct_examples(query_dp, example_pool, 4)
        if distinct_examples:
            train_dataset.append(build_few_shot_training_prompt(query_dp, distinct_examples))
        else:
            # Fallback to zero-shot if can't find enough distinct examples
            train_dataset.append(build_zero_shot_training_prompt(query_dp))
    
    # Shuffle all training prompts together (zero-shot, 2-shot, and 4-shot mixed)
    random.shuffle(train_dataset)
    
    # Count zero-shot vs few-shot (few-shot contains the separator text)
    separator = "\nNow, here is a question"
    num_zero_shot_actual = sum(1 for dp in train_dataset if separator not in dp['text'])
    num_few_shot_actual = len(train_dataset) - num_zero_shot_actual
    
    # Count 2-shot vs 4-shot (count examples in each prompt)
    num_two_shot_actual = 0
    num_four_shot_actual = 0
    for dp in train_dataset:
        if separator in dp['text']:
            # Count how many "Example" strings appear before the separator
            text_before_sep = dp['text'].split(separator)[0]
            example_count = text_before_sep.count("Example ")
            if example_count == 2:
                num_two_shot_actual += 1
            elif example_count == 4:
                num_four_shot_actual += 1
    
    print(f"✅ Built {len(train_dataset)} training prompts")
    print(f"   Zero-shot: {num_zero_shot_actual}")
    print(f"   2-shot: {num_two_shot_actual}")
    print(f"   4-shot: {num_four_shot_actual}")
    
    # Generate validation samples (same parity constraint, size 6 only)
    print(f"\n--- VALIDATION SAMPLES (size {VAL_NODE_SIZE} only) ---")
    print(f"   Generating {VAL_SAMPLES_TOTAL} validation samples")
    print(f"   Task: Find shortest path from start to end only using nodes of the same parity")
    print(f"   Constraint: Path must only use nodes with same parity (even/odd) as start node")
    print(f"   Evaluation: 0-shot, 2-shot, and 4-shot")
    print(f"   Breakdown: {VAL_QUERIES} queries + 100 examples (2-shot) + 200 examples (4-shot) = {VAL_SAMPLES_TOTAL} total\n")
    
    val_samples = []
    for i in range(VAL_SAMPLES_TOTAL):
        if (i + 1) % PRINT_PROGRESS_INTERVAL == 0:
            print(f"  Progress: {i + 1}/{VAL_SAMPLES_TOTAL}")
        
        datapoint = generate_validation_datapoint(VAL_NODE_SIZE, seen_graph_hashes, allow_used_graphs=False)
        
        if datapoint is None:
            print(f"  Warning: Failed to generate unique datapoint {i + 1}")
            continue
        
        val_samples.append(datapoint)
        # Mark graph as used
        graph_hash = hashlib.md5(str(datapoint["adjacency_matrix"]).encode()).hexdigest()
        seen_graph_hashes.add(graph_hash)
    
    print(f"✅ Generated {len(val_samples)} validation samples")
    print(f"   Unique graphs: {len(seen_graph_hashes)}")
    
    # Convert validation to instruction/input/output format (for callback compatibility)
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
    print("IMPORTANT: This is TRUE ICL setup with SAME PARITY evaluation (0, 2, 4-shot)")
    print("  TRAIN ON SIZES 3-9, TEST ON SIZE 6")
    print("  - Training: 1/3 zero-shot, 1/3 2-shot, 1/3 4-shot (equal proportions, shuffled together)")
    print(f"  - Training: Sizes {TRAIN_NODE_SIZES} - {len(train_datapoints)} datapoints")
    print("  - Training format: Uses inference format (not alpaca)")
    print(f"  - Evaluation: Size {VAL_NODE_SIZE} only - {VAL_QUERIES} queries")
    print("  - Evaluation format: 'Find shortest path from node X to node Y only using nodes of the same parity'")
    print("  - Constraint: Path must only use nodes with same parity (even/odd) as start node")
    print("  - Evaluation modes: 0-shot, 2-shot, and 4-shot")
    print(f"  - Total completions: {VAL_QUERIES} queries × 3 modes = {VAL_QUERIES * 3} completions")
    print("  - Model must learn the NEW task format from examples at inference time")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

