use clap::Parser;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct Graph {
    n: usize,
    adj: Vec<Vec<(usize, i32)>>, // (neighbor, weight)
}

impl Graph {
    fn new(n: usize) -> Self {
        Graph {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    fn add_edge(&mut self, u: usize, v: usize, weight: i32) {
        self.adj[u].push((v, weight));
        self.adj[v].push((u, weight));
    }

    fn shortest_path_length(&self, start: usize, end: usize) -> Option<i32> {
        if start == end {
            return Some(0);
        }

        let mut dist = vec![i32::MAX; self.n];
        dist[start] = 0;
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((0, start)));

        while let Some(Reverse((d, u))) = heap.pop() {
            if u == end {
                return Some(d);
            }
            if d > dist[u] {
                continue;
            }
            for &(v, w) in &self.adj[u] {
                let new_dist = d + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    heap.push(Reverse((new_dist, v)));
                }
            }
        }

        None
    }

    fn to_adjacency_matrix(&self) -> Vec<Vec<i32>> {
        let mut matrix = vec![vec![0; self.n]; self.n];
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                matrix[u][v] = w;
            }
        }
        matrix
    }
}

fn generate_erdos_renyi_graph(
    n: usize,
    p: f64,
    min_weight: i32,
    max_weight: i32,
    rng: &mut StdRng,
) -> (Vec<Vec<i32>>, Graph) {
    let mut graph = Graph::new(n);

    // Generate Erdős–Rényi graph
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                let weight = rng.gen_range(min_weight..=max_weight);
                graph.add_edge(u, v, weight);
            }
        }
    }

    (graph.to_adjacency_matrix(), graph)
}

#[derive(Serialize, Deserialize)]
struct Example {
    u: usize,
    v: usize,
    length: i32,
}

#[derive(Serialize, Deserialize)]
struct Query {
    u: usize,
    v: usize,
    length: i32,
}

#[derive(Serialize, Deserialize)]
struct Datapoint {
    adjacency_matrix: Vec<Vec<i32>>,
    num_nodes: usize,
    examples: Vec<Example>,
    query: Query,
}

fn generate_datapoint(
    num_nodes: usize,
    edge_probability: f64,
    min_weight: i32,
    max_weight: i32,
    num_examples: usize,
    max_unique_attempts: usize,
    rng: &mut StdRng,
) -> Option<Datapoint> {
    for _graph_attempt in 0..max_unique_attempts {
        // Generate graph
        let (adj_matrix, graph) = generate_erdos_renyi_graph(
            num_nodes,
            edge_probability,
            min_weight,
            max_weight,
            rng,
        );

        // Track used pairs
        let mut used_pairs = HashSet::new();
        let mut examples = Vec::new();
        let mut example_attempts = 0;

        // Generate unique examples
        while examples.len() < num_examples && example_attempts < max_unique_attempts {
            example_attempts += 1;
            
            let u = rng.gen_range(0..num_nodes);
            let v = rng.gen_range(0..num_nodes);
            
            if u == v {
                continue;
            }

            let pair_key = if u < v { (u, v) } else { (v, u) };
            if used_pairs.contains(&pair_key) {
                continue;
            }

            if let Some(length) = graph.shortest_path_length(u, v) {
                used_pairs.insert(pair_key);
                examples.push(Example {
                    u: u + 1, // Convert to 1-indexed
                    v: v + 1, // Convert to 1-indexed
                    length,
                });
            }
        }

        if examples.len() < num_examples {
            continue;
        }

        // Generate query pair
        let mut query_attempts = 0;
        let mut query_result = None;

        while query_attempts < max_unique_attempts {
            query_attempts += 1;
            
            let u_q = rng.gen_range(0..num_nodes);
            let v_q = rng.gen_range(0..num_nodes);
            
            if u_q == v_q {
                continue;
            }

            let query_pair_key = if u_q < v_q { (u_q, v_q) } else { (v_q, u_q) };
            if used_pairs.contains(&query_pair_key) {
                continue;
            }

            if let Some(query_length) = graph.shortest_path_length(u_q, v_q) {
                query_result = Some(Query {
                    u: u_q + 1, // Convert to 1-indexed
                    v: v_q + 1, // Convert to 1-indexed
                    length: query_length,
                });
                break;
            }
        }

        if let Some(query) = query_result {
            return Some(Datapoint {
                adjacency_matrix: adj_matrix,
                num_nodes,
                examples,
                query,
            });
        }
    }

    None
}

fn get_datapoint_hash(datapoint: &Datapoint) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    datapoint.adjacency_matrix.hash(&mut hasher);
    serde_json::to_string(&datapoint.examples).unwrap().hash(&mut hasher);
    serde_json::to_string(&datapoint.query).unwrap().hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn convert_to_qwen_format(datapoint: &Datapoint) -> serde_json::Value {
    let num_nodes = datapoint.num_nodes;
    let adj_matrix = &datapoint.adjacency_matrix;
    
    // Extract vertices list
    let vertices: Vec<usize> = (1..=num_nodes).collect();
    let vertices_str = vertices.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
    
    // Extract edges (for undirected graph, only list each edge once: u < v)
    let mut edges = Vec::new();
    for u in 0..num_nodes {
        for v in (u + 1)..num_nodes {
            let weight = adj_matrix[u][v];
            if weight > 0 {
                edges.push((u + 1, v + 1, weight));
            }
        }
    }
    
    // Sort edges for consistent output
    edges.sort();
    
    // Build graph section matching example_prompt.txt format
    let mut graph_parts = vec![
        "Given the following undirected weighted graph:".to_string(),
        "".to_string(),
        "Vertices:".to_string(),
        vertices_str,
        "".to_string(),
        "Edge Weights:".to_string(),
    ];
    
    for (u, v, weight) in edges {
        graph_parts.push(format!("{} {}: {}", u, v, weight));
    }
    
    // Add task description
    graph_parts.push("".to_string());
    graph_parts.push("Task: For each pair u v, output the length of the shortest path between u and v.".to_string());
    
    // Add "Shortest path length:" header
    graph_parts.push("".to_string());
    graph_parts.push("Shortest path length:".to_string());
    
    // Add example pairs in "Example:" format with "Input:" and "Output:"
    for example in &datapoint.examples {
        graph_parts.push("Example:".to_string());
        graph_parts.push(format!("Input: {} {}", example.u, example.v));
        graph_parts.push(format!("Output: {}", example.length));
        graph_parts.push("".to_string());
    }
    
    // Add query pair in "Input:" and "Output: " format (without answer)
    let query = &datapoint.query;
    graph_parts.push(format!("Input: {} {}", query.u, query.v));
    graph_parts.push("Output: ".to_string());
    
    let input_text = graph_parts.join("\n");
    let output_text = query.length.to_string();
    
    serde_json::json!({
        "input": input_text,
        "output": output_text
    })
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate synthetic shortest path length datasets using Erdős–Rényi random graphs",
    long_about = "Generate synthetic datasets for in-context learning of shortest path problems.\n\
                  Creates graphs from Erdős–Rényi G(n, p) distribution with weighted edges.\n\
                  Each datapoint contains a graph, example shortest path pairs, and a query pair.\n\
                  Outputs train and validation JSON files in instruction format."
)]
struct Args {
    /// Number of datapoints to generate
    #[arg(short = 'N', long, default_value_t = 500000)]
    num_datapoints: usize,

    /// Edge probability p for Erdős–Rényi G(n, p) distribution
    #[arg(short = 'p', long, default_value_t = 0.3)]
    edge_probability: f64,

    /// Train split ratio (fraction of data for training, remainder goes to validation).
    /// For example, 0.1 means 10% train / 90% val, 0.8 means 80% train / 20% val.
    #[arg(short = 's', long, default_value_t = 0.1)]
    train_split: f64,

    /// Minimum number of nodes (inclusive)
    #[arg(long, default_value_t = 1)]
    min_nodes: usize,

    /// Maximum number of nodes (inclusive)
    #[arg(long, default_value_t = 30)]
    max_nodes: usize,

    /// Minimum edge weight
    #[arg(long, default_value_t = 1)]
    min_weight: i32,

    /// Maximum edge weight
    #[arg(long, default_value_t = 50)]
    max_weight: i32,

    /// Number of example pairs to include
    #[arg(short = 'e', long, default_value_t = 5)]
    num_examples: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 182)]
    seed: u64,

    /// Output directory for train and val JSON files
    #[arg(short = 'o', long, default_value = "data")]
    output_dir: PathBuf,

    /// Train output filename
    #[arg(long, default_value = "train_len.json")]
    train_output: String,

    /// Val output filename
    #[arg(long, default_value = "val_len.json")]
    val_output: String,

    /// Maximum attempts for rejection sampling per pair
    #[arg(long, default_value_t = 100)]
    max_rejection_attempts: usize,

    /// Maximum attempts to generate unique examples before regenerating graph
    #[arg(long, default_value_t = 500)]
    max_unique_attempts: usize,

    /// Print progress every N datapoints
    #[arg(long, default_value_t = 100)]
    print_progress_interval: usize,
}

fn main() {
    let args = Args::parse();

    // Validate arguments
    if args.train_split <= 0.0 || args.train_split >= 1.0 {
        eprintln!("Error: train_split must be between 0 and 1 (exclusive)");
        std::process::exit(1);
    }
    if args.edge_probability < 0.0 || args.edge_probability > 1.0 {
        eprintln!("Error: edge_probability must be between 0 and 1");
        std::process::exit(1);
    }
    if args.min_nodes > args.max_nodes {
        eprintln!("Error: min_nodes must be <= max_nodes");
        std::process::exit(1);
    }
    if args.min_weight > args.max_weight {
        eprintln!("Error: min_weight must be <= max_weight");
        std::process::exit(1);
    }

    // Initialize RNG
    let mut rng = StdRng::seed_from_u64(args.seed);

    let num_nodes_list: Vec<usize> = (args.min_nodes..=args.max_nodes).collect();

    println!("Generating {} unique datapoints...", args.num_datapoints);
    println!("Graph parameters: Erdős–Rényi G(n, p) with n in [{}, {}], p = {}", 
             args.min_nodes, args.max_nodes, args.edge_probability);
    println!("Edge weights: integers in [{}, {}]", args.min_weight, args.max_weight);
    println!("In-context learning: {} examples + 1 query per datapoint", args.num_examples);
    println!("Train/Val split: {:.1}%/{:.1}%", 
             args.train_split * 100.0, (1.0 - args.train_split) * 100.0);

    let mut datapoints = Vec::new();
    let mut seen_hashes = HashSet::new();
    let mut attempts = 0;
    let max_attempts = args.num_datapoints * 20; // Prevent infinite loops

    while datapoints.len() < args.num_datapoints && attempts < max_attempts {
        attempts += 1;
        
        // Randomly select number of nodes from num_nodes_list
        let num_nodes = num_nodes_list[rng.gen_range(0..num_nodes_list.len())];
        
        let datapoint = generate_datapoint(
            num_nodes,
            args.edge_probability,
            args.min_weight,
            args.max_weight,
            args.num_examples,
            args.max_unique_attempts,
            &mut rng,
        );
        
        // Skip if generation failed
        if let Some(dp) = datapoint {
            // Check for duplicates
            let datapoint_hash = get_datapoint_hash(&dp);
            if !seen_hashes.contains(&datapoint_hash) {
                seen_hashes.insert(datapoint_hash);
                datapoints.push(dp);
                
                if datapoints.len() % args.print_progress_interval == 0 {
                    println!("Generated {}/{} unique datapoints...", 
                             datapoints.len(), args.num_datapoints);
                }
            }
        }
    }

    if datapoints.len() < args.num_datapoints {
        println!("Warning: Only generated {} unique datapoints out of {} requested.",
                 datapoints.len(), args.num_datapoints);
    }

    // Shuffle datapoints before splitting
    use rand::seq::SliceRandom;
    datapoints.shuffle(&mut rng);

    // Split into train and val
    // train_split is the fraction that goes to training (e.g., 0.1 = 10% train, 90% val)
    let split_idx = (datapoints.len() as f64 * args.train_split) as usize;
    let train_datapoints = &datapoints[..split_idx];
    let val_datapoints = &datapoints[split_idx..];

    // Convert to instruction format
    let train_dataset: Vec<serde_json::Value> = train_datapoints
        .iter()
        .map(convert_to_qwen_format)
        .collect();
    
    let val_dataset: Vec<serde_json::Value> = val_datapoints
        .iter()
        .map(convert_to_qwen_format)
        .collect();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    // Save train dataset
    let train_path = args.output_dir.join(&args.train_output);
    let train_file = std::fs::File::create(&train_path)
        .expect("Failed to create train output file");
    serde_json::to_writer(train_file, &train_dataset)
        .expect("Failed to write train dataset");

    // Save val dataset
    let val_path = args.output_dir.join(&args.val_output);
    let val_file = std::fs::File::create(&val_path)
        .expect("Failed to create val output file");
    serde_json::to_writer(val_file, &val_dataset)
        .expect("Failed to write val dataset");

    println!("Train dataset saved to {:?} ({} datapoints)", train_path, train_datapoints.len());
    println!("Val dataset saved to {:?} ({} datapoints)", val_path, val_datapoints.len());
    println!("Total unique datapoints: {}", datapoints.len());
    println!("Train/Val split: {}/{} ({:.1}%/{:.1}%)",
             train_datapoints.len(), val_datapoints.len(),
             args.train_split * 100.0, (1.0 - args.train_split) * 100.0);
}

