#!/usr/bin/env python3
"""
Generate dataset for HIDDEN RULE learning pipeline.

TRAINING:
- Always 4-shot prompts
- Hidden rules: 50% "maximum edge weight" (random max weight per prompt), 50% "no adjacent index edges" (cannot use edges from i to i+1 or i+1 to i)
- All 4 examples in a prompt share the same hidden rule
- Don't tell model the rule explicitly
- Model must infer rule from examples
- true path != constrained path
- Size 6 graphs

EVALUATION:
- Hidden rule: "avoid some node" (random node, not stated)
- 0-shot or 4-shot at runtime
- Same queries for 0-shot and 4-shot
- true path != constrained path
- Size 6 graphs
- Format matches training format
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

NODE_SIZE = 6  # Both training and evaluation use size 6

# Number of samples
TRAIN_PROMPTS = 10000  # 10,000 training prompts (each has 4 examples + 1 query = 5 datapoints)
VAL_QUERIES = 50  # 50 queries (same for 0-shot and 4-shot)
VAL_EXAMPLES = 200  # 200 examples for 4-shot evaluation (50 queries * 4 examples each)

SEED = 182
TRAIN_OUTPUT_FILE = "data/train_hidden_rule_max_edge_weight_4shot_size6_256tokens.json"
VAL_OUTPUT_FILE = "data/val_hidden_rule_avoid_0shot_4shot_size6_256tokens.json"

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


def dijkstra_standard(adjacency_matrix: np.ndarray, start: int, end: int) -> Tuple[Optional[List[int]], int]:
    """Compute standard shortest path without any constraints."""
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


def dijkstra_max_edge_weight_with_trace(adjacency_matrix: np.ndarray, start: int, end: int, max_edge_weight: int) -> Tuple[Optional[List[int]], int, List[str]]:
    """
    Compute shortest path where no edge has weight > max_edge_weight.
    
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
    reasoning.append(f"Maximum allowed edge weight: {max_edge_weight}")
    
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
                
                # Only consider edges with weight <= max_edge_weight
                if edge_weight > max_edge_weight:
                    continue
                
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    neighbors_explored.append(f"node {neighbor} (dist: {new_dist}, edge: {edge_weight})")
        
        if neighbors_explored:
            reasoning.append(f"Updated distances for: {', '.join(neighbors_explored)}")
    
    reasoning.append(f"No path found from {start} to {end} with all edges having weight <= {max_edge_weight}")
    return None, -1, reasoning


def dijkstra_no_adjacent_index_with_trace(adjacency_matrix: np.ndarray, start: int, end: int) -> Tuple[Optional[List[int]], int, List[str]]:
    """
    Compute shortest path that avoids edges between adjacent index nodes.
    Cannot use edges from i to i+1 or i+1 to i.
    
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
    reasoning.append("Cannot use edges between adjacent index nodes (i to i+1 or i+1 to i)")
    
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
                # Check if this edge is between adjacent indices
                # Edge from i to i+1 or i+1 to i is forbidden
                if abs(current - neighbor) == 1:
                    continue  # Skip adjacent index edges
                
                edge_weight = adjacency_matrix[current][neighbor]
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    neighbors_explored.append(f"node {neighbor} (dist: {new_dist})")
        
        if neighbors_explored:
            reasoning.append(f"Updated distances for: {', '.join(neighbors_explored)}")
    
    reasoning.append(f"No path found from {start} to {end} without using adjacent index edges")
    return None, -1, reasoning


def dijkstra_avoid_node_with_trace(adjacency_matrix: np.ndarray, start: int, end: int, avoid_node: int) -> Tuple[Optional[List[int]], int, List[str]]:
    """Compute shortest path that avoids avoid_node."""
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    visited = [False] * n
    reasoning = []
    
    reasoning.append(f"Starting at node {start} with distance 0")
    reasoning.append(f"Must avoid node {avoid_node}")
    
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
            if neighbor == avoid_node:
                continue
                
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
    
    reasoning.append(f"No path found from {start} to {end} that avoids node {avoid_node}")
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


def generate_training_datapoint_max_edge_weight(num_nodes: int, max_edge_weight: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate a single training datapoint with max edge weight constraint."""
    max_attempts = 100 * num_nodes  # Reasonable attempts
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        graph_hash = hashlib.md5(str(adjacency.tolist()).encode()).hexdigest()
        if graph_hash in seen_graph_hashes:
            continue
        
        # Pick random start and end nodes (faster than computing all reachable)
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        
        if start_node == end_node:
            continue
        
        # Check standard path first (quick check)
        standard_path, standard_distance = dijkstra_standard(adjacency, start_node, end_node)
        
        if standard_path is None or standard_distance < 0:
            continue
        
        # Quick check: does standard path violate constraint?
        standard_violates = False
        for i in range(len(standard_path) - 1):
            edge_weight = adjacency[standard_path[i]][standard_path[i + 1]]
            if edge_weight > max_edge_weight:
                standard_violates = True
                break
        
        if not standard_violates:
            continue
        
        # Find constrained path (max edge weight)
        path, distance, reasoning_steps = dijkstra_max_edge_weight_with_trace(
            adjacency, start_node, end_node, max_edge_weight
        )
        
        if path is None or distance < 0:
            continue
        
        # CRITICAL: Ensure standard path ≠ constrained path
        if standard_path == path:
            continue
        
        # Remove the "Maximum allowed edge weight" line from reasoning to hide the rule
        filtered_reasoning = [r for r in reasoning_steps if not r.startswith("Maximum allowed edge weight:")]
        
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(filtered_reasoning)
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
            "max_edge_weight": max_edge_weight,  # Use the constraint (may have been adjusted)
            "rule_type": "max_edge_weight",  # Mark the rule type
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": filtered_reasoning if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        seen_graph_hashes.add(graph_hash)
        return datapoint
    
    return None


def generate_eval_datapoint_avoid_node(num_nodes: int, avoid_node: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate evaluation datapoint with avoid node constraint."""
    max_attempts = 100 * num_nodes  # Reasonable attempts
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        graph_hash = hashlib.md5(str(adjacency.tolist()).encode()).hexdigest()
        if graph_hash in seen_graph_hashes:
            continue
        
        # Pick random start and end nodes (faster)
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        
        if start_node == end_node or start_node == avoid_node or end_node == avoid_node:
            continue
        
        # Find shortest path that avoids the avoid node
        path, distance, reasoning_steps = dijkstra_avoid_node_with_trace(
            adjacency, start_node, end_node, avoid_node
        )
        
        if path is None or distance < 0:
            continue
        
        # Check standard path
        standard_path, standard_distance = dijkstra_standard(adjacency, start_node, end_node)
        
        if standard_path is None or standard_distance < 0:
            continue
        
        # CRITICAL: Ensure standard path ≠ constrained path
        if standard_path == path:
            continue
        
        # Remove the "Must avoid node" line from reasoning to hide the rule
        filtered_reasoning = [r for r in reasoning_steps if not r.startswith("Must avoid node")]
        
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(filtered_reasoning)
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
            "avoid_node": avoid_node,
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": filtered_reasoning if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        seen_graph_hashes.add(graph_hash)
        return datapoint
    
    return None


def generate_training_datapoint_no_adjacent_index(num_nodes: int, seen_graph_hashes: set) -> Optional[dict]:
    """Generate a single training datapoint with no adjacent index edges constraint."""
    max_attempts = 100 * num_nodes  # Reasonable attempts
    
    for attempt in range(max_attempts):
        adjacency = generate_random_graph(num_nodes)
        
        graph_hash = hashlib.md5(str(adjacency.tolist()).encode()).hexdigest()
        if graph_hash in seen_graph_hashes:
            continue
        
        # Pick random start and end nodes (faster)
        start_node = random.randint(0, num_nodes - 1)
        end_node = random.randint(0, num_nodes - 1)
        
        if start_node == end_node:
            continue
        
        # Check standard path first
        standard_path, standard_distance = dijkstra_standard(adjacency, start_node, end_node)
        
        if standard_path is None or standard_distance < 0:
            continue
        
        # Quick check: does standard path use adjacent index edges?
        standard_uses_adjacent = False
        for i in range(len(standard_path) - 1):
            if abs(standard_path[i] - standard_path[i + 1]) == 1:
                standard_uses_adjacent = True
                break
        
        if not standard_uses_adjacent:
            # Standard path doesn't use adjacent edges - skip this graph
            # We need graphs where standard path uses adjacent edges
            continue
        
        # Find constrained path (no adjacent index edges)
        path, distance, reasoning_steps = dijkstra_no_adjacent_index_with_trace(
            adjacency, start_node, end_node
        )
        
        if path is None or distance < 0:
            continue
        
        # CRITICAL: Ensure standard path ≠ constrained path
        if standard_path == path:
            continue
        
        # Remove the "Cannot use edges between adjacent index nodes" line from reasoning to hide the rule
        filtered_reasoning = [r for r in reasoning_steps if not r.startswith("Cannot use edges between adjacent index nodes")]
        
        if USE_CHAIN_OF_THOUGHT:
            reasoning_text = "\n".join(filtered_reasoning)
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
            "rule_type": "no_adjacent_index",  # Mark the rule type
            "shortest_path": path,
            "total_distance": distance,
            "reasoning_steps": filtered_reasoning if USE_CHAIN_OF_THOUGHT else [],
            "output": output_text
        }
        
        seen_graph_hashes.add(graph_hash)
        return datapoint
    
    return None


def build_4shot_training_prompt(examples: List[dict], query: dict) -> dict:
    """Build 4-shot training prompt with hidden max edge weight rule."""
    parts = []
    
    parts.append("Below are 4 examples. Study them carefully to infer the hidden rule, then apply it to solve the question.\n")
    
    for idx, example in enumerate(examples):
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
    
    parts.append("\nNow, here is a question for you to solve. Use the examples above to infer the hidden rule and apply it:\n")
    
    query_adjacency = json.dumps(query["adjacency_matrix"])
    parts.append(
        f"Question:\n"
        f"Graph representation: {query_adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {query['start_node']} to node {query['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "Study the examples above to infer the hidden rule, then apply it to solve this question.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    prompt_text = "\n".join(parts)
    
    result = {
        "text": prompt_text + "\n" + query.get("output", ""),
        "adjacency_matrix": query["adjacency_matrix"],
        "num_nodes": query["num_nodes"],
        "start_node": query["start_node"],
        "end_node": query["end_node"],
        "shortest_path": query["shortest_path"],
        "total_distance": query["total_distance"],
        "reasoning_steps": query.get("reasoning_steps", []),
    }
    
    # Add rule-specific fields
    if "max_edge_weight" in query:
        result["max_edge_weight"] = query["max_edge_weight"]
        result["rule_type"] = "max_edge_weight"
    elif "rule_type" in query:
        result["rule_type"] = query["rule_type"]
    
    return result


def build_eval_prompt_0shot(datapoint: dict) -> dict:
    """Build 0-shot evaluation prompt (avoid node rule, not stated)."""
    adjacency = json.dumps(datapoint["adjacency_matrix"])
    
    instruction = (
        f"Graph representation: {adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {datapoint['start_node']} to node {datapoint['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "Study the examples (if provided) to infer the hidden rule, then apply it to solve this question.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' "
        "where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    return {
        "instruction": instruction,
        "input": "",
        "output": datapoint.get("output", ""),
        "adjacency_matrix": datapoint["adjacency_matrix"],
        "num_nodes": datapoint["num_nodes"],
        "start_node": datapoint["start_node"],
        "end_node": datapoint["end_node"],
        "avoid_node": datapoint["avoid_node"],
        "shortest_path": datapoint["shortest_path"],
        "total_distance": datapoint["total_distance"],
        "reasoning_steps": datapoint.get("reasoning_steps", []),
    }


def build_eval_prompt_4shot(examples: List[dict], query: dict) -> dict:
    """Build 4-shot evaluation prompt (avoid node rule, not stated)."""
    parts = []
    
    parts.append("Below are 4 examples. Study them carefully to infer the hidden rule, then apply it to solve the question.\n")
    
    for idx, example in enumerate(examples):
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
    
    parts.append("\nNow, here is a question for you to solve. Use the examples above to infer the hidden rule and apply it:\n")
    
    query_adjacency = json.dumps(query["adjacency_matrix"])
    parts.append(
        f"Question:\n"
        f"Graph representation: {query_adjacency}\n"
        f"This is an adjacency matrix where 0 means no edge exists, and a positive number represents the edge weight.\n"
        f"Find the shortest path from node {query['start_node']} to node {query['end_node']}.\n"
        "Nodes are indexed from 0.\n"
        "Study the examples above to infer the hidden rule, then apply it to solve this question.\n"
        "You must end your response with 'Final Answer: [path] with a total distance of [total]' where [path] is the shortest path as a list of node indices and [total] is the total distance."
    )
    
    prompt_text = "\n".join(parts)
    
    return {
        "instruction": prompt_text,
        "input": "",
        "output": query.get("output", ""),
        "adjacency_matrix": query["adjacency_matrix"],
        "num_nodes": query["num_nodes"],
        "start_node": query["start_node"],
        "end_node": query["end_node"],
        "avoid_node": query["avoid_node"],
        "shortest_path": query["shortest_path"],
        "total_distance": query["total_distance"],
        "reasoning_steps": query.get("reasoning_steps", []),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    
    print(f"\n{'='*80}")
    print("HIDDEN RULE LEARNING PIPELINE")
    print("TRAINING: 4-shot with hidden rules (50% max edge weight, 50% no adjacent index edges)")
    print("EVALUATION: 0-shot/4-shot with hidden 'avoid node' rule")
    print(f"{'='*80}\n")
    
    seen_graph_hashes = set()
    
    # ========================================================================
    # TRAINING DATA GENERATION
    # ========================================================================
    print(f"\n--- TRAINING DATA (4-shot, hidden rules: 50% max edge weight, 50% no adjacent index) ---")
    print(f"   Generating {TRAIN_PROMPTS} training prompts")
    print(f"   Each prompt: 4 examples + 1 query (all share same hidden rule)")
    print(f"   Rules: 50% Maximum edge weight, 50% No adjacent index edges (i to i+1 or i+1 to i)")
    print(f"   Size: {NODE_SIZE} nodes\n")
    
    train_dataset = []
    
    for prompt_idx in range(TRAIN_PROMPTS):
        if (prompt_idx + 1) % PRINT_PROGRESS_INTERVAL == 0:
            print(f"  Progress: {prompt_idx + 1}/{TRAIN_PROMPTS}")
        
        # Alternate between max_edge_weight and no_adjacent_index (50/50 split)
        use_max_edge_weight = (prompt_idx % 2 == 0)
        
        examples = []
        query = None
        max_prompt_retries = 2
        
        for prompt_retry in range(max_prompt_retries):
            examples = []
            query = None
            
            if use_max_edge_weight:
                # Pick a random max edge weight for this prompt
                max_edge_weight = random.randint(MIN_WEIGHT, min(4, MAX_WEIGHT - 1))
                
                for i in range(5):  # 4 examples + 1 query
                    datapoint = generate_training_datapoint_max_edge_weight(NODE_SIZE, max_edge_weight, seen_graph_hashes)
                    
                    if datapoint is None:
                        break
                    
                    if i < 4:
                        examples.append(datapoint)
                    else:
                        query = datapoint
            else:
                # Use no adjacent index rule
                for i in range(5):  # 4 examples + 1 query
                    datapoint = generate_training_datapoint_no_adjacent_index(NODE_SIZE, seen_graph_hashes)
                    
                    if datapoint is None:
                        break
                    
                    if i < 4:
                        examples.append(datapoint)
                    else:
                        query = datapoint
            
            if len(examples) == 4 and query is not None:
                prompt = build_4shot_training_prompt(examples, query)
                train_dataset.append(prompt)
                break
        
        if len(examples) < 4 or query is None:
            if (prompt_idx + 1) % 100 == 0:  # Only print every 100 to avoid spam
                print(f"  Warning: Failed to generate complete prompt {prompt_idx + 1}")
    
    print(f"✅ Generated {len(train_dataset)} training prompts")
    
    # ========================================================================
    # EVALUATION DATA GENERATION
    # ========================================================================
    print(f"\n--- EVALUATION DATA (0-shot/4-shot, hidden avoid node rule) ---")
    print(f"   Generating {VAL_QUERIES} queries")
    print(f"   Generating {VAL_EXAMPLES} examples (for 4-shot)")
    print(f"   Rule: Avoid a random node (not stated)")
    print(f"   Size: {NODE_SIZE} nodes\n")
    
    # Generate queries (50 queries, each with random avoid node)
    val_queries = []
    for i in range(VAL_QUERIES):
        if (i + 1) % 10 == 0:
            print(f"  Query progress: {i + 1}/{VAL_QUERIES}")
        
        avoid_node = random.randint(0, NODE_SIZE - 1)
        datapoint = generate_eval_datapoint_avoid_node(NODE_SIZE, avoid_node, seen_graph_hashes)
        
        if datapoint is None:
            if (i + 1) % 10 == 0:  # Only print every 10 to avoid spam
                print(f"  Warning: Failed to generate query {i + 1}, retrying with different avoid node...")
            # Try a different avoid node
            avoid_node = random.randint(0, NODE_SIZE - 1)
            datapoint = generate_eval_datapoint_avoid_node(NODE_SIZE, avoid_node, seen_graph_hashes)
        
        if datapoint is None:
            continue
        
        val_queries.append(datapoint)
    
    print(f"✅ Generated {len(val_queries)} queries")
    
    # Generate examples (200 examples, each with random avoid node)
    val_examples = []
    for i in range(VAL_EXAMPLES):
        if (i + 1) % 50 == 0:
            print(f"  Example progress: {i + 1}/{VAL_EXAMPLES}")
        
        avoid_node = random.randint(0, NODE_SIZE - 1)
        datapoint = generate_eval_datapoint_avoid_node(NODE_SIZE, avoid_node, seen_graph_hashes)
        
        if datapoint is None:
            # Try a different avoid node once
            avoid_node = random.randint(0, NODE_SIZE - 1)
            datapoint = generate_eval_datapoint_avoid_node(NODE_SIZE, avoid_node, seen_graph_hashes)
        
        if datapoint is None:
            continue
        
        val_examples.append(datapoint)
    
    print(f"✅ Generated {len(val_examples)} examples")
    
    # Build evaluation dataset: same queries for 0-shot and 4-shot
    val_dataset = []
    
    # 0-shot prompts (first 50)
    for query in val_queries:
        val_dataset.append(build_eval_prompt_0shot(query))
    
    # 4-shot prompts (next 50, same queries)
    for query in val_queries:
        # Select 4 random examples (they may have different avoid nodes, that's okay)
        if len(val_examples) >= 4:
            selected_examples = random.sample(val_examples, 4)
            prompt = build_eval_prompt_4shot(selected_examples, query)
            val_dataset.append(prompt)
        else:
            print(f"  Warning: Not enough examples for 4-shot prompts")
    
    print(f"✅ Built {len(val_dataset)} evaluation prompts")
    print(f"   - 0-shot: {len(val_queries)} prompts (indices 0-{len(val_queries)-1})")
    print(f"   - 4-shot: {len(val_dataset) - len(val_queries)} prompts (indices {len(val_queries)}-{len(val_dataset)-1})")
    
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
    print("SUMMARY")
    print(f"  Training: {len(train_dataset)} prompts (4-shot, hidden max edge weight rule)")
    print(f"  Evaluation: {len(val_queries)} queries × 2 modes = {len(val_dataset)} prompts")
    print(f"    - 0-shot: {len(val_queries)} prompts")
    print(f"    - 4-shot: {len(val_dataset) - len(val_queries)} prompts")
    print(f"  All graphs: Size {NODE_SIZE}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

