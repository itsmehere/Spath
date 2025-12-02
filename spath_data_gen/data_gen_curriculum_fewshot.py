#!/usr/bin/env python3
"""
Generate curriculum learning dataset with few-shot format (4 examples + 1 query per batch).
Includes Dijkstra's algorithm reasoning in the output.
"""

import json
import random
import numpy as np
from typing import List, Tuple, Optional
import heapq
import hashlib
from pathlib import Path

# ============================================================================
# CURRICULUM LEARNING CONFIGURATION
# ============================================================================

# Curriculum stages for few-shot training
CURRICULUM_STAGES = [
    {"num_nodes": [3], "samples": 5000, "name": "stage1_baby"},         # 5K samples, 3 nodes (easiest)
    {"num_nodes": [4], "samples": 10000, "name": "stage2_easy"},        # 10K samples, 4 nodes
    {"num_nodes": [5], "samples": 10000, "name": "stage3_medium"},      # 10K samples, 5 nodes
    {"num_nodes": [6], "samples": 15000, "name": "stage4_hard"},       # 15K samples, 6 nodes
    {"num_nodes": [7], "samples": 15000, "name": "stage5_harder"},     # 15K samples, 7 nodes (hardest)
]

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_curriculum_fewshot.json"
VAL_OUTPUT_FILE = "data/val_curriculum_fewshot.json"
TRAIN_SPLIT = 0.98  # 98% train, 2% val

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

# Few-shot configuration
GRAPHS_PER_BATCH = 5  # 4 examples + 1 query

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
            reasoning.append(f"  Updated distances for: {', '.join(neighbors_explored)}")
    
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


def generate_datapoint(num_nodes: int, curriculum_stage: str) -> dict:
    """Generate a single datapoint with adjacency matrix and shortest path."""
    adjacency_matrix = generate_random_graph(num_nodes)
    start_node = random.randint(0, num_nodes - 1)
    
    reachable_nodes = dijkstra_all_reachable(adjacency_matrix, start_node)
    reachable_end_nodes = [node for node in reachable_nodes if node != start_node]
    
    if not reachable_end_nodes:
        return generate_datapoint(num_nodes, curriculum_stage)
    
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
        "curriculum_stage": curriculum_stage,
        "edge_probability": EDGE_PROBABILITY,
    }


def build_fewshot_batch_prompt(batch: List[dict], num_nodes: int) -> dict:
    """Build a few-shot prompt with 4 examples + 1 query, including Dijkstra reasoning."""
    if len(batch) < GRAPHS_PER_BATCH:
        raise ValueError(f"Expected {GRAPHS_PER_BATCH} graphs per batch, got {len(batch)}")
    
    parts = []
    
    # Add 4 examples with answers
    for idx, example in enumerate(batch[:4]):
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
        
        parts.append(
            f"Answer: {json.dumps(example.get('shortest_path', []))}\n"
        )
    
    # Add query (5th example) without answer
    query = batch[4]
    adjacency = json.dumps(query["adjacency_matrix"])
    parts.append(
        f"Example 5:\n"
        f"Graph representation: {adjacency}\n"
        f"Find the shortest path from node {query['start_node']} to node {query['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "Use Dijkstra's algorithm to compute the shortest path for the query graph.\n"
        "Answer: Provide the correct shortest path for the query graph."
    )
    
    prompt_text = "\n".join(parts)
    
    # Build output with Dijkstra reasoning
    if USE_CHAIN_OF_THOUGHT and query.get('reasoning_steps'):
        reasoning_text = "\n".join(query['reasoning_steps'])
        output_text = (
            f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
            f"{reasoning_text}\n\n"
            f"Final Answer: The shortest path from node {query['start_node']} "
            f"to node {query['end_node']} is {json.dumps(query.get('shortest_path', []))} "
            f"with total distance {query.get('total_distance', 'unknown')}."
        )
    else:
        output_text = (
            f"Shortest path: {json.dumps(query.get('shortest_path', []))}\n"
            f"Total distance: {query.get('total_distance', 'unknown')}"
        )
    
    return {
        "instruction": prompt_text,
        "input": "",
        "output": output_text,
        "num_nodes": num_nodes,
        "curriculum_stage": query.get("curriculum_stage"),
    }


def generate_curriculum_fewshot_dataset():
    """Generate curriculum learning dataset with few-shot format."""
    random.seed(SEED)
    np.random.seed(SEED)
    
    all_datapoints = []
    seen_hashes = set()
    
    print(f"Generating curriculum few-shot dataset with {len(CURRICULUM_STAGES)} stages...")
    print(f"Configuration: edge_prob={EDGE_PROBABILITY}, weights={MIN_WEIGHT}-{MAX_WEIGHT}")
    print(f"Chain-of-thought: {USE_CHAIN_OF_THOUGHT}")
    print(f"Graphs per batch: {GRAPHS_PER_BATCH} (4 examples + 1 query)\n")
    
    # Generate data for each curriculum stage
    for stage_idx, stage in enumerate(CURRICULUM_STAGES):
        stage_name = stage["name"]
        num_nodes_list = stage["num_nodes"]
        target_samples = stage["samples"]
        
        print(f"{'='*80}")
        print(f"STAGE {stage_idx + 1}: {stage_name}")
        print(f"  Nodes: {num_nodes_list}")
        print(f"  Target samples: {target_samples}")
        print(f"{'='*80}")
        
        stage_datapoints = []
        attempts = 0
        max_attempts = target_samples * 10
        
        while len(stage_datapoints) < target_samples and attempts < max_attempts:
            attempts += 1
            
            num_nodes = random.choice(num_nodes_list)
            datapoint = generate_datapoint(num_nodes, stage_name)
            
            datapoint_hash = hashlib.md5(
                (str(datapoint['adjacency_matrix']) + 
                 str(datapoint['start_node']) + 
                 str(datapoint['end_node'])).encode()
            ).hexdigest()
            
            if datapoint_hash not in seen_hashes:
                seen_hashes.add(datapoint_hash)
                datapoint['id'] = len(all_datapoints)
                datapoint['stage_id'] = len(stage_datapoints)
                stage_datapoints.append(datapoint)
                all_datapoints.append(datapoint)
                
                if len(stage_datapoints) % PRINT_PROGRESS_INTERVAL == 0:
                    print(f"  Generated {len(stage_datapoints)}/{target_samples} samples...")
        
        print(f"  ✓ Completed stage {stage_idx + 1}: {len(stage_datapoints)} samples\n")
    
    total_samples = len(all_datapoints)
    print(f"\n{'='*80}")
    print(f"TOTAL: Generated {total_samples} unique datapoints across all stages")
    print(f"{'='*80}\n")
    
    # Split into train and val WHILE PRESERVING CURRICULUM ORDER
    split_idx = int(len(all_datapoints) * TRAIN_SPLIT)
    train_datapoints = all_datapoints[:split_idx]
    val_datapoints = all_datapoints[split_idx:]
    
    # Group into batches of 5 (4 examples + 1 query) per stage
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
    print(f"Stage distribution in training set:")
    for stage in CURRICULUM_STAGES:
        stage_name = stage["name"]
        count = sum(1 for batch in train_batches if batch.get("curriculum_stage") == stage_name)
        percentage = 100 * count / len(train_batches) if train_batches else 0
        print(f"  {stage_name}: {count} batches ({percentage:.1f}%)")
    
    print(f"\n✓ Train dataset: {train_path} ({len(train_batches)} batches, {len(train_datapoints)} graphs)")
    print(f"✓ Val dataset: {val_path} ({len(val_batches)} batches, {len(val_datapoints)} graphs)")
    print(f"✓ Split: {100*TRAIN_SPLIT:.1f}% train / {100*(1-TRAIN_SPLIT):.1f}% val")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: Training data is in CURRICULUM ORDER!")
    print("Easy examples (3 nodes) come first, harder examples (7 nodes) come later.")
    print("Do NOT shuffle the training data!")
    print(f"{'='*80}\n")
    
    return train_batches, val_batches


if __name__ == "__main__":
    generate_curriculum_fewshot_dataset()

