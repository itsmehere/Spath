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

# Dataset generation parameters for curriculum learning
# Total: 100K samples split across 6 stages from 3 to 15 nodes
# Distribution: 15% / 15% / 20% / 20% / 15% / 15%
CURRICULUM_STAGES = [
    {"num_nodes": [3, 4], "samples": 15000, "name": "stage1_baby"},        # 15K samples, 3-4 nodes (15%)
    {"num_nodes": [5, 6], "samples": 15000, "name": "stage2_easy"},        # 15K samples, 5-6 nodes (15%)
    {"num_nodes": [7, 8], "samples": 20000, "name": "stage3_medium"},      # 20K samples, 7-8 nodes (20%)
    {"num_nodes": [9, 10], "samples": 20000, "name": "stage4_hard"},       # 20K samples, 9-10 nodes (20%)
    {"num_nodes": [11, 12], "samples": 15000, "name": "stage5_harder"},    # 15K samples, 11-12 nodes (15%)
    {"num_nodes": [13, 14, 15], "samples": 15000, "name": "stage6_expert"},# 15K samples, 13-15 nodes (15%)
]

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_curriculum.json"
VAL_OUTPUT_FILE = "data/val_curriculum.json"
TRAIN_SPLIT = 0.98  # 98% train, 2% val

# Graph generation parameters
EDGE_PROBABILITY = 0.4
MIN_WEIGHT = 1
MAX_WEIGHT = 10

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
    """Generate a random directed graph as an adjacency matrix."""
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < EDGE_PROBABILITY:
                adjacency_matrix[i][j] = random.randint(MIN_WEIGHT, MAX_WEIGHT)
    
    return adjacency_matrix


def get_datapoint_hash(datapoint: dict) -> str:
    """Create a hash for duplicate detection."""
    adj_tuple = tuple(tuple(row) for row in datapoint['adjacency_matrix'])
    unique_str = f"{adj_tuple}_{datapoint['start_node']}_{datapoint['end_node']}"
    return hashlib.md5(unique_str.encode()).hexdigest()


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
        "curriculum_stage": curriculum_stage,  # Track which stage this is from
    }


def convert_to_qwen_format(datapoints: List[dict]) -> List[dict]:
    """Convert datapoints to instruction-input-output format with chain-of-thought."""
    dataset = []
    for datapoint in datapoints:
        adj_matrix_str = json.dumps(datapoint['adjacency_matrix'])
        
        input_text = (
            f"Graph representation: {adj_matrix_str}\n"
            f"This is a {datapoint['num_nodes']}x{datapoint['num_nodes']} adjacency matrix "
            f"where 0 means no edge and positive numbers represent edge weights. "
            f"Nodes are indexed from 0 to {datapoint['num_nodes']-1}."
        )
        
        instruction = (
            f"In this directed graph, find the shortest path from node {datapoint['start_node']} "
            f"to node {datapoint['end_node']} using Dijkstra's algorithm. "
            f"Show your reasoning step by step."
        )
        
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(datapoint['reasoning_steps'])
            output_text = (
                f"I'll find the shortest path using Dijkstra's algorithm:\n\n"
                f"{reasoning_text}\n\n"
                f"Final Answer: The shortest path from node {datapoint['start_node']} "
                f"to node {datapoint['end_node']} is {json.dumps(datapoint['shortest_path'])} "
                f"with total distance {datapoint['total_distance']}."
            )
        else:
            output_text = (
                f"Shortest path: {json.dumps(datapoint['shortest_path'])}, "
                f"Distance: {datapoint['total_distance']}"
            )
        
        formatted_datapoint = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "curriculum_stage": datapoint["curriculum_stage"],  # Include stage info
        }
        dataset.append(formatted_datapoint)
    
    return dataset


def generate_curriculum_dataset():
    """Generate curriculum learning dataset with progressive difficulty."""
    random.seed(SEED)
    np.random.seed(SEED)
    
    all_datapoints = []
    seen_hashes = set()
    
    print(f"Generating curriculum learning dataset with {len(CURRICULUM_STAGES)} stages...")
    print(f"Configuration: edge_prob={EDGE_PROBABILITY}, weights={MIN_WEIGHT}-{MAX_WEIGHT}")
    print(f"Chain-of-thought: {USE_CHAIN_OF_THOUGHT}\n")
    
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
            
            datapoint_hash = get_datapoint_hash(datapoint)
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
    # This is crucial - we want train data in curriculum order
    split_idx = int(len(all_datapoints) * TRAIN_SPLIT)
    train_datapoints = all_datapoints[:split_idx]
    val_datapoints = all_datapoints[split_idx:]
    
    # Update IDs
    for idx, dp in enumerate(train_datapoints):
        dp['train_id'] = idx
    for idx, dp in enumerate(val_datapoints):
        dp['val_id'] = idx
    
    # Convert to instruction format
    train_dataset = convert_to_qwen_format(train_datapoints)
    val_dataset = convert_to_qwen_format(val_datapoints)
    
    # Save train dataset (preserve order for curriculum learning!)
    train_path = Path(__file__).parent / TRAIN_OUTPUT_FILE
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w') as f:
        json.dump(train_dataset, f, indent=2)
    
    val_path = Path(__file__).parent / VAL_OUTPUT_FILE
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w') as f:
        json.dump(val_dataset, f, indent=2)
    
    # Print stage distribution
    print(f"Stage distribution in training set:")
    for stage in CURRICULUM_STAGES:
        stage_name = stage["name"]
        count = sum(1 for dp in train_dataset if dp["curriculum_stage"] == stage_name)
        percentage = 100 * count / len(train_dataset)
        print(f"  {stage_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✓ Train dataset: {train_path} ({len(train_datapoints)} samples)")
    print(f"✓ Val dataset: {val_path} ({len(val_datapoints)} samples)")
    print(f"✓ Split: {100*TRAIN_SPLIT:.1f}% train / {100*(1-TRAIN_SPLIT):.1f}% val")
    
    print(f"\n{'='*80}")
    print("IMPORTANT: Training data is in CURRICULUM ORDER!")
    print("Easy examples (3-4 nodes) come first, harder examples (6-8 nodes) come later.")
    print("Do NOT shuffle the training data!")
    print(f"{'='*80}\n")
    
    return train_datapoints, val_datapoints


if __name__ == "__main__":
    generate_curriculum_dataset()

