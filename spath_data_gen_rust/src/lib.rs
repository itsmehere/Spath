use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

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

#[pyfunction]
fn generate_erdos_renyi_graph(
    n: usize,
    p: f64,
    min_weight: i32,
    max_weight: i32,
    seed: Option<u64>,
) -> PyResult<(Vec<Vec<i32>>, Vec<(usize, usize, i32)>)> {
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let mut graph = Graph::new(n);
    let mut edges = Vec::new();

    // Generate Erdős–Rényi graph
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                let weight = rng.gen_range(min_weight..=max_weight);
                graph.add_edge(u, v, weight);
                edges.push((u, v, weight));
            }
        }
    }

    Ok((graph.to_adjacency_matrix(), edges))
}

#[pyfunction]
fn shortest_path_length_from_matrix(
    adj_matrix: Vec<Vec<i32>>,
    start: usize,
    end: usize,
) -> PyResult<Option<i32>> {
    let n = adj_matrix.len();
    let mut graph = Graph::new(n);
    
    for u in 0..n {
        for v in 0..n {
            if adj_matrix[u][v] > 0 {
                graph.add_edge(u, v, adj_matrix[u][v]);
            }
        }
    }

    Ok(graph.shortest_path_length(start, end))
}

#[pyfunction]
fn sample_vertex_pair_with_path(
    adj_matrix: Vec<Vec<i32>>,
    max_attempts: usize,
    seed: Option<u64>,
) -> PyResult<Option<(usize, usize, i32)>> {
    let n = adj_matrix.len();
    if n < 2 {
        return Ok(None);
    }

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

    let graph = {
        let mut g = Graph::new(n);
        for u in 0..n {
            for v in 0..n {
                if adj_matrix[u][v] > 0 {
                    g.add_edge(u, v, adj_matrix[u][v]);
                }
            }
        }
        g
    };

    for _ in 0..max_attempts {
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);
        
        if u == v {
            continue;
        }

        if let Some(length) = graph.shortest_path_length(u, v) {
            return Ok(Some((u, v, length)));
        }
    }

    Ok(None)
}

fn generate_erdos_renyi_graph_internal(
    n: usize,
    p: f64,
    min_weight: i32,
    max_weight: i32,
    seed: u64,
) -> (Vec<Vec<i32>>, Vec<(usize, usize, i32)>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph = Graph::new(n);
    let mut edges = Vec::new();

    // Generate Erdős–Rényi graph
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen::<f64>() < p {
                let weight = rng.gen_range(min_weight..=max_weight);
                graph.add_edge(u, v, weight);
                edges.push((u, v, weight));
            }
        }
    }

    (graph.to_adjacency_matrix(), edges)
}

#[pyfunction]
fn generate_datapoint_rust(
    num_nodes: usize,
    edge_probability: f64,
    min_weight: i32,
    max_weight: i32,
    num_examples: usize,
    max_rejection_attempts: usize,
    max_unique_attempts: usize,
    seed: Option<u64>,
) -> PyResult<Option<PyObject>> {
    Python::with_gil(|py| {
        let mut rng = if let Some(s) = seed {
            StdRng::seed_from_u64(s)
        } else {
            StdRng::from_entropy()
        };

        for _graph_attempt in 0..max_unique_attempts {
            // Generate graph
            let (adj_matrix, _edges) = generate_erdos_renyi_graph_internal(
                num_nodes,
                edge_probability,
                min_weight,
                max_weight,
                rng.gen(),
            );

            let graph = {
                let mut g = Graph::new(num_nodes);
                for u in 0..num_nodes {
                    for v in 0..num_nodes {
                        if adj_matrix[u][v] > 0 {
                            g.add_edge(u, v, adj_matrix[u][v]);
                        }
                    }
                }
                g
            };

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
                    examples.push((u, v, length));
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
                    query_result = Some((u_q, v_q, query_length));
                    break;
                }
            }

            if let Some((u_q, v_q, query_length)) = query_result {
                // Convert to Python objects
                let py_examples: Vec<PyObject> = examples
                    .iter()
                    .map(|(u, v, len)| {
                        let dict = PyDict::new_bound(py);
                        dict.set_item("u", u + 1).unwrap();
                        dict.set_item("v", v + 1).unwrap();
                        dict.set_item("length", *len).unwrap();
                        dict.into()
                    })
                    .collect();

                let py_query: PyObject = {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("u", u_q + 1).unwrap();
                    dict.set_item("v", v_q + 1).unwrap();
                    dict.set_item("length", query_length).unwrap();
                    dict.into()
                };

                let result = PyDict::new_bound(py);
                result.set_item("adjacency_matrix", adj_matrix).unwrap();
                result.set_item("num_nodes", num_nodes).unwrap();
                result.set_item("examples", py_examples).unwrap();
                result.set_item("query", py_query).unwrap();

                return Ok(Some(result.into()));
            }
        }

        Ok(None)
    })
}

#[pymodule]
fn spath_data_gen_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_erdos_renyi_graph, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path_length_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(sample_vertex_pair_with_path, m)?)?;
    m.add_function(wrap_pyfunction!(generate_datapoint_rust, m)?)?;
    Ok(())
}

