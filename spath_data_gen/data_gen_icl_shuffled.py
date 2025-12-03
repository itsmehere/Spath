#!/usr/bin/env python3
"""
Generate ICL (in-context learning) dataset with shuffled data (no curriculum).
5 ICL examples + 1 query per batch.
Includes Dijkstra's algorithm reasoning in the output.
Train on sizes: 3, 4, 6, 8, 9, 10
Val on sizes: 5, 7
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
TRAIN_NODE_SIZES = [3, 4, 6, 8, 9, 10]
VAL_NODE_SIZES = [5, 7]

# Number of samples per node size
SAMPLES_PER_SIZE = 10000  # Adjust as needed
VAL_SAMPLES_PER_SIZE = 50  # Validation samples per size

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_icl_5_shuffled.json"
VAL_OUTPUT_FILE = "data/val_icl_5_shuffled.json"

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

# Few-shot configuration
GRAPHS_PER_BATCH = 6  # 5 examples + 1 query

PRINT_PROGRESS_INTERVAL = 1000
USE_CHAIN_OF_THOUGHT = True


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
    """Compute shortest path with reasoning trace for chain-of-thought."""
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
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = previous[node]
            path = path[::-1]
            
            reasoning.append(f"Reached destination node {end}")
            reasoning.append(f"Reconstructed path: {path}")
            
            return path, int(current_dist), reasoning
        
        neighbors_explored = []
        for neighbor in range(n):
            if not visited[neighbor] and adjacency_matrix[current][neighbor] > 0:
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    neighbors_explored.append(f"node {neighbor} (dist: {new_dist})")
        
        if neighbors_explored:
            reasoning.append(f"Updated distances for: {', '.join(neighbors_explored)}")
    
    return None, -1, reasoning + ["No path found"]


def generate_random_graph(num_nodes: int) -> np.ndarray:
    """Generate a random directed graph as an adjacency matrix (no negative edges)."""
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < EDGE_PROBABILITY:
                # Only positive weights, no negative edges
                adjacency_matrix[i][j] = random.randint(MIN_WEIGHT, MAX_WEIGHT)
    
    return adjacency_matrix


def generate_datapoint(num_nodes: int) -> dict:
    """Generate a single datapoint with adjacency matrix and shortest path."""
    adjacency_matrix = generate_random_graph(num_nodes)
    start_node = random.randint(0, num_nodes - 1)
    
    reachable_nodes = dijkstra_all_reachable(adjacency_matrix, start_node)
    reachable_end_nodes = [node for node in reachable_nodes if node != start_node]
    
    if not reachable_end_nodes:
        return generate_datapoint(num_nodes)
    
    end_node = random.choice(reachable_end_nodes)
    
    shortest_path, total_distance, reasoning_steps = dijkstra_with_trace(
        adjacency_matrix, start_node, end_node
    )
    
    return {
        "adjacency_matrix": adjacency_matrix.tolist(),
        "shortest_path": shortest_path,
        "total_distance": total_distance,
        "reasoning_steps": reasoning_steps,
        "num_nodes": num_nodes,
        "start_node": start_node,
        "end_node": end_node,
        "edge_probability": EDGE_PROBABILITY,
    }


def build_fewshot_batch_prompt(batch: List[dict], num_nodes: int) -> dict:
    """Build a few-shot prompt with 5 examples + 1 query, including Dijkstra reasoning."""
    if len(batch) < GRAPHS_PER_BATCH:
        raise ValueError(f"Expected {GRAPHS_PER_BATCH} graphs per batch, got {len(batch)}")
    
    parts = []
    
    # Add 5 examples with answers
    for idx, example in enumerate(batch[:5]):
        adjacency = json.dumps(example["adjacency_matrix"])
        parts.append(
            f"Example {idx + 1}:\n"
            f"Graph representation: {adjacency}\n"
            f"Find the shortest path from node {example['start_node']} to node {example['end_node']}.\n"
            "Nodes are indexed from 0.\n"
        )
        
        # Add Dijkstra reasoning for examples
        if USE_CHAIN_OF_THOUGHT and example.get('reasoning_steps'):
            reasoning_text = "\n".join(example['reasoning_steps'])
            parts.append(
                f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
                f"{reasoning_text}\n\n"
            )
    
    # Add query (6th example) without answer
    query = batch[5]
    adjacency = json.dumps(query["adjacency_matrix"])
    
    prompt_text = "\n".join(parts)
    
    # Build output with Dijkstra reasoning
    if USE_CHAIN_OF_THOUGHT and query.get('reasoning_steps'):
        reasoning_text = "\n".join(query['reasoning_steps'])
        output_text = (
            f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
            f"{reasoning_text}\n\n"
            f"Final Answer: The shortest path from node {query['start_node']} to node {query['end_node']} is {json.dumps(query.get('shortest_path', []))} with total distance {query.get('total_distance', 'unknown')}."
        )
    else:
        output_text = (
            f"Shortest path: {json.dumps(query.get('shortest_path', []))}\n"
            f"Total distance: {query.get('total_distance', 'unknown')}"
        )
    
    return {
        "instruction": f"Example 6:\nGraph representation: {adjacency}\nFind the shortest path from node {query['start_node']} to node {query['end_node']}.\nNodes are indexed from 0.\n",
        "input": prompt_text,
        "output": output_text,
        "num_nodes": num_nodes,
    }


def generate_icl_shuffled_dataset():
    """Generate ICL dataset with shuffled data (no curriculum)."""
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"Generating ICL shuffled dataset...")
    print(f"Train node sizes: {TRAIN_NODE_SIZES}")
    print(f"Val node sizes: {VAL_NODE_SIZES}")
    print(f"Configuration: edge_prob={EDGE_PROBABILITY}, weights={MIN_WEIGHT}-{MAX_WEIGHT}")
    print(f"Chain-of-thought: {USE_CHAIN_OF_THOUGHT}")
    print(f"Graphs per batch: {GRAPHS_PER_BATCH} (5 examples + 1 query)\n")
    
    # Generate training datapoints
    train_datapoints = []
    seen_hashes = set()
    
    print(f"{'='*80}")
    print("GENERATING TRAINING DATA")
    print(f"{'='*80}")
    
    for num_nodes in TRAIN_NODE_SIZES:
        print(f"\nGenerating {SAMPLES_PER_SIZE} samples for {num_nodes} nodes...")
        node_datapoints = []
        attempts = 0
        max_attempts = SAMPLES_PER_SIZE * 10
        
        while len(node_datapoints) < SAMPLES_PER_SIZE and attempts < max_attempts:
            attempts += 1
            
            datapoint = generate_datapoint(num_nodes)
            
            datapoint_hash = hashlib.md5(
                (str(datapoint['adjacency_matrix']) + 
                 str(datapoint['start_node']) + 
                 str(datapoint['end_node'])).encode()
            ).hexdigest()
            
            if datapoint_hash not in seen_hashes:
                seen_hashes.add(datapoint_hash)
                datapoint['id'] = len(train_datapoints)
                node_datapoints.append(datapoint)
                train_datapoints.append(datapoint)
                
                if len(node_datapoints) % PRINT_PROGRESS_INTERVAL == 0:
                    print(f"  Generated {len(node_datapoints)}/{SAMPLES_PER_SIZE} samples...")
        
        print(f"  ✓ Completed {num_nodes} nodes: {len(node_datapoints)} samples")
    
    print(f"\n✓ Total training datapoints: {len(train_datapoints)}")
    
    # Generate validation datapoints
    val_datapoints = []
    
    print(f"\n{'='*80}")
    print("GENERATING VALIDATION DATA")
    print(f"{'='*80}")
    
    for num_nodes in VAL_NODE_SIZES:
        print(f"\nGenerating {VAL_SAMPLES_PER_SIZE} samples for {num_nodes} nodes...")
        node_datapoints = []
        attempts = 0
        max_attempts = VAL_SAMPLES_PER_SIZE * 10
        
        while len(node_datapoints) < VAL_SAMPLES_PER_SIZE and attempts < max_attempts:
            attempts += 1
            
            datapoint = generate_datapoint(num_nodes)
            
            datapoint_hash = hashlib.md5(
                (str(datapoint['adjacency_matrix']) + 
                 str(datapoint['start_node']) + 
                 str(datapoint['end_node'])).encode()
            ).hexdigest()
            
            if datapoint_hash not in seen_hashes:
                seen_hashes.add(datapoint_hash)
                datapoint['id'] = len(val_datapoints)
                node_datapoints.append(datapoint)
                val_datapoints.append(datapoint)
                
                if len(node_datapoints) % 10 == 0:
                    print(f"  Generated {len(node_datapoints)}/{VAL_SAMPLES_PER_SIZE} samples...")
        
        print(f"  ✓ Completed {num_nodes} nodes: {len(node_datapoints)} samples")
    
    print(f"\n✓ Total validation datapoints: {len(val_datapoints)}")
    
    # Verify node sizes
    train_sizes = set(dp['num_nodes'] for dp in train_datapoints)
    val_sizes = set(dp['num_nodes'] for dp in val_datapoints)
    
    assert train_sizes == set(TRAIN_NODE_SIZES), f"Train sizes mismatch: expected {set(TRAIN_NODE_SIZES)}, got {train_sizes}"
    assert val_sizes == set(VAL_NODE_SIZES), f"Val sizes mismatch: expected {set(VAL_NODE_SIZES)}, got {val_sizes}"
    
    # SHUFFLE the data (no curriculum ordering)
    print(f"\n{'='*80}")
    print("SHUFFLING DATA (no curriculum ordering)")
    print(f"{'='*80}")
    random.shuffle(train_datapoints)
    random.shuffle(val_datapoints)
    print("✓ Data shuffled")
    
    # Group into batches of 6 (5 examples + 1 query)
    def create_batches(datapoints: List[dict]) -> List[dict]:
        batches = []
        for i in range(0, len(datapoints), GRAPHS_PER_BATCH):
            batch_graphs = datapoints[i:i + GRAPHS_PER_BATCH]
            if len(batch_graphs) == GRAPHS_PER_BATCH:
                # Use the num_nodes from the first graph in the batch
                num_nodes = batch_graphs[0]['num_nodes']
                batch_entry = build_fewshot_batch_prompt(batch_graphs, num_nodes)
                batch_entry['batch_id'] = len(batches)
                batches.append(batch_entry)
        return batches
    
    train_batches = create_batches(train_datapoints)
    val_batches = create_batches(val_datapoints)
    
    # Save datasets
    train_path = Path(__file__).parent / TRAIN_OUTPUT_FILE
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_batches, f, indent=2)
    
    val_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_batches, f, indent=2)
    
    # Print statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nTraining set node size distribution:")
    for size in TRAIN_NODE_SIZES:
        count = sum(1 for batch in train_batches if batch.get("num_nodes") == size)
        percentage = 100 * count / len(train_batches) if train_batches else 0
        print(f"  {size} nodes: {count} batches ({percentage:.1f}%)")
    
    print(f"\nValidation set node size distribution:")
    for size in VAL_NODE_SIZES:
        count = sum(1 for batch in val_batches if batch.get("num_nodes") == size)
        percentage = 100 * count / len(val_batches) if val_batches else 0
        print(f"  {size} nodes: {count} batches ({percentage:.1f}%)")
    
    print(f"\n✓ Train dataset: {train_path} ({len(train_batches)} batches, {len(train_datapoints)} graphs)")
    print(f"✓ Val dataset: {val_path} ({len(val_batches)} batches, {len(val_datapoints)} graphs)")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: Data is SHUFFLED (no curriculum ordering)")
    print("You can shuffle the training data during training if desired.")
    print(f"{'='*80}\n")
    
    return train_batches, val_batches


if __name__ == "__main__":
    generate_icl_shuffled_dataset()

