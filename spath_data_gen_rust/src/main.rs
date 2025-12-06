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

#[derive(Clone, Serialize, Deserialize)]
struct Example {
    u: usize,
    v: usize,
    length: i32,
}

#[derive(Clone, Serialize, Deserialize)]
struct Query {
    u: usize,
    v: usize,
    length: i32,
}

#[derive(Clone, Serialize, Deserialize)]
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

#[derive(Clone, Debug)]
struct CurriculumStage {
    num_nodes: Vec<usize>,
    samples: usize,
    name: String,
    min_weight: i32,
    max_weight: i32,
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Generate synthetic shortest path length datasets using Erdős–Rényi random graphs",
    long_about = "Generate synthetic datasets for in-context learning of shortest path problems.\n\
                  Creates graphs from Erdős–Rényi G(n, p) distribution with weighted edges.\n\
                  Each datapoint contains a graph, example shortest path pairs, and a query pair.\n\
                  Outputs train and validation JSON files in instruction format.\n\
                  Supports curriculum learning with progressive difficulty based on graph vertex count."
)]
struct Args {
    /// Number of datapoints to generate (ignored if --use-curriculum is set)
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

    /// Enable curriculum learning with progressive difficulty based on graph vertex count
    #[arg(long, default_value_t = true)]
    use_curriculum: bool,

    /// Number of samples per curriculum stage
    #[arg(long, default_value_t = 3000)]
    samples_per_stage: usize,

    /// Ratio of rehearsal samples from previous stages (0.1 = 10%)
    #[arg(long, default_value_t = 0.1)]
    rehearsal_ratio: f64,

    /// Maximum edge weight cap for scaling (used in get_scaled_max_weight)
    #[arg(long, default_value_t = 50)]
    max_weight_cap: i32,
}

fn get_scaled_max_weight(n: usize, max_weight_cap: i32) -> i32 {
    // Linear scaling: start at 10 for n=5, increase by 2 per node, cap at max_weight_cap
    // Formula: max_weight = min(max_weight_cap, 10 + (n - 5) * 2)
    let calculated = 10 + (n as i32 - 5) * 2;
    calculated.min(max_weight_cap)
}

fn get_default_curriculum_stages(
    min_nodes: usize,
    max_nodes: usize,
    num_examples: usize,
    samples_per_stage: usize,
    min_weight: i32,
    max_weight_cap: i32,
) -> Vec<CurriculumStage> {
    // Create curriculum stages: start with n = k (num_examples) and gradually increase by 1
    // Edge weights also scale with n: max_weight = 10 + (n - 5) * 2 (from 10 at n=5 to max_weight_cap)
    // Each stage uses a single node count
    let mut stages = Vec::new();
    
    // Start from n = k (num_examples), but ensure it's at least min_nodes
    let start_node = num_examples.max(min_nodes);
    
    // Generate stages from start_node to max_nodes, incrementing by 1
    let mut stage_num = 1;
    for n in start_node..=max_nodes {
        let max_weight = get_scaled_max_weight(n, max_weight_cap);
        stages.push(CurriculumStage {
            num_nodes: vec![n],
            samples: samples_per_stage,
            name: format!("stage{}_n{}", stage_num, n),
            min_weight,
            max_weight,
        });
        stage_num += 1;
    }
    
    stages
}

fn generate_curriculum_dataset(args: &Args, rng: &mut StdRng) {
    let curriculum_stages = get_default_curriculum_stages(
        args.min_nodes,
        args.max_nodes,
        args.num_examples,
        args.samples_per_stage,
        args.min_weight,
        args.max_weight_cap,
    );
    let total_samples: usize = curriculum_stages.iter().map(|s| s.samples).sum();
    let rehearsal_ratio = args.rehearsal_ratio;
    
    println!("Generating curriculum learning dataset with {} stages...", curriculum_stages.len());
    println!("Graph parameters: Erdős–Rényi G(n, p) with n in [{}, {}], p = {}", 
             args.min_nodes, args.max_nodes, args.edge_probability);
    println!("Edge weights: scaling with graph size (from 1-10 at n=5 to 1-50 at n=30)");
    println!("In-context learning: {} examples + 1 query per datapoint", args.num_examples);
    println!("Rehearsal ratio: {:.1}% (adding {:.1}% rehearsal samples from previous stages)", 
             rehearsal_ratio * 100.0, rehearsal_ratio * 100.0);
    println!("Train/Val split: {:.1}%/{:.1}%", 
             args.train_split * 100.0, (1.0 - args.train_split) * 100.0);
    println!("Total target samples: {}\n", total_samples);
    
    let mut all_datapoints = Vec::new();
    let mut seen_hashes = HashSet::new();
    let mut previous_stages_datapoints: Vec<Datapoint> = Vec::new(); // Track datapoints from previous stages for rehearsal
    
    // Generate data for each curriculum stage
    for (stage_idx, stage) in curriculum_stages.iter().enumerate() {
        println!("{}", "=".repeat(80));
        println!("STAGE {}: {}", stage_idx + 1, stage.name);
        println!("  Nodes: {:?}", stage.num_nodes);
        println!("  Edge weights: [{}, {}]", stage.min_weight, stage.max_weight);
        println!("  Target samples: {}", stage.samples);
        println!("{}", "=".repeat(80));
        
        let mut stage_datapoints = Vec::new();
        let mut attempts = 0;
        let max_attempts = stage.samples * 20;
        
        while stage_datapoints.len() < stage.samples && attempts < max_attempts {
            attempts += 1;
            
            // Select number of nodes from this stage's range
            let num_nodes = stage.num_nodes[rng.gen_range(0..stage.num_nodes.len())];
            
            let datapoint = generate_datapoint(
                num_nodes,
                args.edge_probability,
                stage.min_weight,
                stage.max_weight,
                args.num_examples,
                args.max_unique_attempts,
                rng,
            );
            
            // Skip if generation failed
            if let Some(dp) = datapoint {
                // Check for duplicates
                let datapoint_hash = get_datapoint_hash(&dp);
                if !seen_hashes.contains(&datapoint_hash) {
                    seen_hashes.insert(datapoint_hash);
                    let dp_clone = dp.clone();
                    stage_datapoints.push(dp_clone.clone());
                    all_datapoints.push(dp_clone);
                    
                    if stage_datapoints.len() % args.print_progress_interval == 0 {
                        println!("  Generated {}/{} samples...", 
                                 stage_datapoints.len(), stage.samples);
                    }
                }
            }
        }
        
        if stage_datapoints.len() < stage.samples {
            println!("  Warning: Only generated {}/{} samples for this stage.", 
                     stage_datapoints.len(), stage.samples);
        }
        
        // Add rehearsal samples from previous stages (if not first stage)
        let mut rehearsal_samples = Vec::new();
        if stage_idx > 0 && !previous_stages_datapoints.is_empty() {
            let num_rehearsal = (stage.samples as f64 * rehearsal_ratio) as usize;
            if num_rehearsal > 0 {
                // Sample random examples from previous stages
                use rand::seq::SliceRandom;
                let num_to_sample = num_rehearsal.min(previous_stages_datapoints.len());
                let mut rehearsal_indices: Vec<usize> = (0..previous_stages_datapoints.len()).collect();
                rehearsal_indices.shuffle(rng);
                
                for &idx in rehearsal_indices.iter().take(num_to_sample) {
                    let rehearsal_dp = previous_stages_datapoints[idx].clone();
                    rehearsal_samples.push(rehearsal_dp);
                }
                
                println!("  Added {} rehearsal samples from previous stages", rehearsal_samples.len());
            }
        }
        
        // Combine stage samples and rehearsal samples
        let num_rehearsal = rehearsal_samples.len();
        let num_new = stage_datapoints.len();
        stage_datapoints.extend(rehearsal_samples.clone());
        all_datapoints.extend(rehearsal_samples);
        
        // Update previous_stages_datapoints for next stage's rehearsal (only non-rehearsal samples)
        previous_stages_datapoints.extend(stage_datapoints.iter().take(num_new).cloned());
        
        println!("  ✓ Completed stage {}: {} total samples ({} new + {} rehearsal)\n", 
                 stage_idx + 1, stage_datapoints.len(), 
                 num_new, num_rehearsal);
    }
    
    let total_generated = all_datapoints.len();
    println!("\n{}", "=".repeat(80));
    println!("TOTAL: Generated {} unique datapoints", total_generated);
    println!("{}", "=".repeat(80));
    println!();
    
    // Split into train and val
    // Training set: preserve curriculum order (easier examples first)
    // Validation set: randomly sample from entire dataset
    let num_val_samples = ((all_datapoints.len() as f64) * (1.0 - args.train_split)) as usize;
    
    // Randomly sample validation indices from entire dataset
    use rand::seq::SliceRandom;
    let mut all_indices: Vec<usize> = (0..all_datapoints.len()).collect();
    all_indices.shuffle(rng);
    let val_indices_set: std::collections::HashSet<usize> = all_indices[..num_val_samples].iter().cloned().collect();
    
    // Split datapoints: train keeps curriculum order, val is random
    let mut train_datapoints = Vec::new();
    let mut val_datapoints = Vec::new();
    
    for (idx, dp) in all_datapoints.iter().enumerate() {
        if val_indices_set.contains(&idx) {
            val_datapoints.push(dp.clone());
        } else {
            train_datapoints.push(dp.clone());
        }
    }
    
    // Shuffle validation set (it's already random, but shuffle for good measure)
    val_datapoints.shuffle(rng);
    
    // Convert to instruction format
    let train_dataset: Vec<serde_json::Value> = train_datapoints
        .iter()
        .map(|dp| convert_to_qwen_format(dp))
        .collect();
    
    let val_dataset: Vec<serde_json::Value> = val_datapoints
        .iter()
        .map(|dp| convert_to_qwen_format(dp))
        .collect();
    
    // Create output directory
    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    
    // Save train dataset (preserve order for curriculum learning!)
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
    
    println!("✓ Train dataset: {:?} ({} datapoints)", train_path, train_datapoints.len());
    println!("✓ Val dataset: {:?} ({} datapoints)", val_path, val_datapoints.len());
    println!("✓ Split: {:.1}%/{:.1}%",
             args.train_split * 100.0, (1.0 - args.train_split) * 100.0);
    
    println!("\n{}", "=".repeat(80));
    println!("IMPORTANT: Training data is in CURRICULUM ORDER!");
    println!("Easy examples (smaller graphs) come first, harder examples (larger graphs) come later.");
    println!("Do NOT shuffle the training data!");
    println!("\nValidation set is RANDOMLY sampled from the entire dataset.");
    println!("{}", "=".repeat(80));
    println!();
}

fn generate_random_dataset(args: &Args, rng: &mut StdRng) {
    let num_nodes_list: Vec<usize> = (args.min_nodes..=args.max_nodes).collect();
    
    println!("Generating {} unique datapoints (random sampling)...", args.num_datapoints);
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
            rng,
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
    datapoints.shuffle(rng);
    
    // Split into train and val
    // train_split is the fraction that goes to training (e.g., 0.1 = 10% train, 90% val)
    let split_idx = (datapoints.len() as f64 * args.train_split) as usize;
    let train_datapoints = &datapoints[..split_idx];
    let val_datapoints = &datapoints[split_idx..];
    
    // Convert to instruction format
    let train_dataset: Vec<serde_json::Value> = train_datapoints
        .iter()
        .map(|dp| convert_to_qwen_format(dp))
        .collect();
    
    let val_dataset: Vec<serde_json::Value> = val_datapoints
        .iter()
        .map(|dp| convert_to_qwen_format(dp))
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

    if args.use_curriculum {
        generate_curriculum_dataset(&args, &mut rng);
    } else {
        generate_random_dataset(&args, &mut rng);
    }
}

