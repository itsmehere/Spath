import json
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
from pathlib import Path

DATA_PATH = "data/val.json"

# Load the dataset
def load_dataset(json_path):
    """Load the synthetic shortest path dataset from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def parse_qwen_datapoint(datapoint):
    """Parse an instruction format datapoint to extract graph information."""
    # Check if it's the new format (instruction/input/output) or old format (conversations)
    if 'instruction' in datapoint:
        # New format
        instruction = datapoint['instruction']
        input_text = datapoint.get('input', '')
        output = datapoint['output']
        
        # In the new format, adjacency matrix is in 'input', question is in 'instruction'
        # Extract adjacency matrix from input field
        input_clean = input_text.replace('\n', ' ')
        matrix_start = input_clean.find('[')
        if matrix_start == -1:
            raise ValueError("Could not find adjacency matrix in input")
        
        # Find the matching closing bracket for the outer array
        bracket_count = 0
        matrix_end = matrix_start
        for i, char in enumerate(input_clean[matrix_start:], start=matrix_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    matrix_end = i + 1
                    break
        
        adj_matrix_str = input_clean[matrix_start:matrix_end]
        try:
            adj_matrix = json.loads(adj_matrix_str)
        except json.JSONDecodeError:
            raise ValueError("Could not parse adjacency matrix from input")
        
        # Extract num_nodes from input (e.g., "15x15")
        num_nodes_match = re.search(r'(\d+)x\d+', input_text)
        if num_nodes_match:
            num_nodes = int(num_nodes_match.group(1))
        else:
            # Fallback: infer from matrix size
            num_nodes = len(adj_matrix)
        
        # Extract start_node and end_node from instruction (e.g., "from node 6 to node 7")
        path_match = re.search(r'from node (\d+) to node (\d+)', instruction)
        if path_match:
            start_node = int(path_match.group(1))
            end_node = int(path_match.group(2))
        else:
            raise ValueError("Could not find start and end nodes in instruction")
            
    elif 'conversations' in datapoint:
        # Old format (for backward compatibility)
        conversations = datapoint['conversations']
        instruction = None
        output = None
        for conv in conversations:
            if conv['from'] == 'human':
                instruction = conv['value']
            elif conv['from'] == 'gpt':
                output = conv['value']
        if instruction is None or output is None:
            raise ValueError("Could not find instruction or output in conversations")
        
        # Extract adjacency matrix from instruction (old format had everything in instruction)
        prompt_clean = instruction.replace('\n', ' ')
        matrix_start = prompt_clean.find('[')
        if matrix_start == -1:
            raise ValueError("Could not find adjacency matrix in instruction")
        
        # Find the matching closing bracket for the outer array
        bracket_count = 0
        matrix_end = matrix_start
        for i, char in enumerate(prompt_clean[matrix_start:], start=matrix_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    matrix_end = i + 1
                    break
        
        adj_matrix_str = prompt_clean[matrix_start:matrix_end]
        try:
            adj_matrix = json.loads(adj_matrix_str)
        except json.JSONDecodeError:
            raise ValueError("Could not parse adjacency matrix from instruction")
        
        # Extract num_nodes from prompt (e.g., "15x15")
        num_nodes_match = re.search(r'(\d+)x\d+', instruction)
        if num_nodes_match:
            num_nodes = int(num_nodes_match.group(1))
        else:
            # Fallback: infer from matrix size
            num_nodes = len(adj_matrix)
        
        # Extract start_node and end_node from prompt (e.g., "from node 6 to node 7")
        path_match = re.search(r'from node (\d+) to node (\d+)', instruction)
        if path_match:
            start_node = int(path_match.group(1))
            end_node = int(path_match.group(2))
        else:
            raise ValueError("Could not find start and end nodes in instruction")
    else:
        raise ValueError("Unknown datapoint format")
    
    # Extract shortest path from output (e.g., "[6, 10, 2, 7]")
    path_match = re.search(r'\[[\d,\s]+\]', output)
    if path_match:
        shortest_path_str = path_match.group(0)
        shortest_path = json.loads(shortest_path_str)
    else:
        raise ValueError("Could not find shortest path in output")
    
    # Get ID if available, otherwise use index
    datapoint_id = datapoint.get('id', 0)
    
    return {
        'id': datapoint_id,
        'adjacency_matrix': adj_matrix,
        'shortest_path': shortest_path,
        'num_nodes': num_nodes,
        'start_node': start_node,
        'end_node': end_node
    }

def visualize_datapoint(datapoint, ax_matrix, ax_graph):
    """Visualize a single datapoint with adjacency matrix and graph."""
    # Parse the instruction format datapoint
    parsed = parse_qwen_datapoint(datapoint)
    
    adj_matrix = np.array(parsed['adjacency_matrix'])
    shortest_path = parsed['shortest_path']
    num_nodes = parsed['num_nodes']
    start_node = parsed['start_node']
    end_node = parsed['end_node']
    datapoint_id = parsed['id']
    
    # Visualize adjacency matrix
    im = ax_matrix.imshow(adj_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=adj_matrix.max() if adj_matrix.max() > 0 else 1)
    ax_matrix.set_title(f'Adjacency Matrix (ID: {datapoint_id})', fontsize=12, fontweight='bold')
    ax_matrix.set_xlabel('To Node', fontsize=10)
    ax_matrix.set_ylabel('From Node', fontsize=10)
    
    # Add text annotations for non-zero values
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] > 0:
                ax_matrix.text(j, i, str(adj_matrix[i][j]), 
                             ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_matrix, label='Edge Weight')
    
    # Visualize graph with shortest path
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(i + 1)  # 1-indexed nodes
    
    # Add edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] > 0:
                G.add_edge(i + 1, j + 1, weight=adj_matrix[i][j])
    
    # Create layout with increased spacing between nodes
    pos = nx.spring_layout(G, k=3.0, iterations=100, seed=42)
    
    # Draw all edges in light gray
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='lightgray', 
                          width=1, alpha=0.5, arrows=True, arrowsize=10)
    
    # Draw shortest path edges in red and bold
    path_edges = []
    if shortest_path and len(shortest_path) > 1:
        for k in range(len(shortest_path) - 1):
            path_edges.append((shortest_path[k], shortest_path[k + 1]))
    
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=path_edges,
                          edge_color='red', width=3, alpha=0.8, arrows=True, arrowsize=15)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == start_node:
            node_colors.append('green')  # Start node in green
        elif node == end_node:
            node_colors.append('red')  # End node in red
        elif node in shortest_path:
            node_colors.append('orange')  # Path nodes in orange
        else:
            node_colors.append('lightblue')  # Other nodes in light blue
    
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors,
                          node_size=500, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=10, font_weight='bold')
    
    # Draw edge labels (weights)
    edge_labels = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] > 0:
                edge_labels[(i + 1, j + 1)] = str(adj_matrix[i][j])
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax_graph, font_size=7)
    
    # Set title
    path_str = ' -> '.join(map(str, shortest_path)) if shortest_path else 'No path'
    ax_graph.set_title(f'Graph (ID: {datapoint_id})\nShortest Path: {path_str}\nStart: {start_node}, End: {end_node}',
                      fontsize=11, fontweight='bold')
    ax_graph.axis('off')

def main():
    # Load dataset
    json_path = Path(__file__).parent / DATA_PATH
    dataset = load_dataset(json_path)
    
    # Randomly select 5 datapoints
    if len(dataset) < 5:
        selected = dataset
        print(f"Warning: Only {len(dataset)} datapoints available, showing all of them.")
    else:
        selected = random.sample(dataset, 5)
    
    # Create output folder
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to {output_dir}")
    
    # Create and save individual images for each datapoint
    for idx, datapoint in enumerate(selected):
        # Create figure with two subplots for each datapoint
        fig, (ax_matrix, ax_graph) = plt.subplots(1, 2, figsize=(16, 6))
        
        visualize_datapoint(datapoint, ax_matrix, ax_graph)
        
        plt.tight_layout()
        
        # Save individual image
        datapoint_id = datapoint.get('id', idx)
        output_path = output_dir / f"datapoint_{datapoint_id}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved datapoint to {output_path}")
        plt.close()  # Close figure to free memory
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()

