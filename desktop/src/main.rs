use dioxus::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

const WIDTH: f32 = 1040.0;
const HEIGHT: f32 = 760.0;
const MAX_RENDERED_EDGES: usize = 1400;

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let graph = build_graph(load_dataset());

    rsx! {
        style { {APP_CSS} }
        main { class: "app",
            section { class: "panel",
                h1 { "What Anime Should I Watch" }
                p { class: "muted", "Desktop Dioxus graph from anonymized user ratings." }
                div { class: "stats",
                    StatRow { label: "Users", value: graph.user_count.to_string() }
                    StatRow { label: "Anime", value: graph.anime_count.to_string() }
                    StatRow { label: "Nodes", value: graph.nodes.len().to_string() }
                    StatRow { label: "Edges (rendered)", value: graph.edges.len().to_string() }
                }
                p { class: "tiny", "For readability, the SVG caps visible edges at 1,400." }
            }
            section { class: "canvas-wrap",
                svg {
                    width: "{WIDTH}",
                    height: "{HEIGHT}",
                    view_box: "0 0 {WIDTH} {HEIGHT}",
                    for edge in graph.edges.iter().take(MAX_RENDERED_EDGES) {
                        line {
                            x1: "{edge.x1}",
                            y1: "{edge.y1}",
                            x2: "{edge.x2}",
                            y2: "{edge.y2}",
                            stroke: "{edge.color}",
                            stroke_width: "{edge.stroke_width}",
                            stroke_opacity: "0.55"
                        }
                    }
                    for node in &graph.nodes {
                        circle {
                            cx: "{node.x}",
                            cy: "{node.y}",
                            r: "{node.radius}",
                            fill: "{node.color}"
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn StatRow(label: String, value: String) -> Element {
    rsx! {
        div { class: "row",
            span { "{label}" }
            strong { "{value}" }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Dataset {
    users: Vec<UserRatings>,
}

#[derive(Debug, Clone, Deserialize)]
struct UserRatings {
    #[serde(rename = "userId")]
    user_id: String,
    ratings: Vec<Rating>,
}

#[derive(Debug, Clone, Deserialize)]
struct Rating {
    #[serde(rename = "animeId")]
    anime_id: u32,
    title: String,
    #[serde(rename = "rawScore")]
    raw_score: f64,
    #[serde(rename = "normalizedScore")]
    normalized_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeType {
    User,
    Anime,
}

#[derive(Debug, Clone)]
struct Node {
    id: String,
    label: String,
    node_type: NodeType,
    x: f32,
    y: f32,
    radius: f32,
    color: &'static str,
}

#[derive(Debug, Clone)]
struct Edge {
    source: usize,
    target: usize,
    color: &'static str,
    stroke_width: f32,
}

#[derive(Debug, Clone)]
struct RenderEdge {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: &'static str,
    stroke_width: f32,
}

#[derive(Debug, Clone)]
struct GraphModel {
    user_count: usize,
    anime_count: usize,
    nodes: Vec<Node>,
    edges: Vec<RenderEdge>,
}

fn load_dataset() -> Dataset {
    let candidates = [
        "../data/anonymized-ratings.json",
        "data/anonymized-ratings.json",
        "../../data/anonymized-ratings.json",
    ];

    for candidate in candidates {
        if let Ok(content) = fs::read_to_string(candidate) {
            if let Ok(dataset) = serde_json::from_str::<Dataset>(&content) {
                return dataset;
            }
        }
    }

    serde_json::from_str(SAMPLE_DATASET).expect("embedded sample dataset is valid JSON")
}

fn build_graph(mut dataset: Dataset) -> GraphModel {
    for user in &mut dataset.users {
        let avg = if user.ratings.is_empty() {
            0.0
        } else {
            user.ratings.iter().map(|r| r.raw_score).sum::<f64>() / user.ratings.len() as f64
        };
        for rating in &mut user.ratings {
            rating.normalized_score = rating.raw_score - avg;
        }
    }

    let mut nodes: Vec<Node> = Vec::new();
    let mut node_index: HashMap<String, usize> = HashMap::new();
    let mut anime_pair_weights: HashMap<(u32, u32), f64> = HashMap::new();
    let mut edges: Vec<Edge> = Vec::new();

    for user in &dataset.users {
        let user_node_id = format!("user:{}", user.user_id);
        let user_idx = upsert_node(
            &mut nodes,
            &mut node_index,
            user_node_id,
            format!("User {}", &user.user_id[..8.min(user.user_id.len())]),
            NodeType::User,
        );

        for rating in &user.ratings {
            let anime_node_id = format!("anime:{}", rating.anime_id);
            let anime_idx = upsert_node(
                &mut nodes,
                &mut node_index,
                anime_node_id,
                rating.title.clone(),
                NodeType::Anime,
            );

            edges.push(Edge {
                source: user_idx,
                target: anime_idx,
                color: "#f4d35ea6",
                stroke_width: 1.5,
            });
        }

        for i in 0..user.ratings.len() {
            for j in (i + 1)..user.ratings.len() {
                let left = &user.ratings[i];
                let right = &user.ratings[j];
                let pair_key = if left.anime_id < right.anime_id {
                    (left.anime_id, right.anime_id)
                } else {
                    (right.anime_id, left.anime_id)
                };
                let pair_score = (left.normalized_score + right.normalized_score) / 2.0;

                anime_pair_weights
                    .entry(pair_key)
                    .and_modify(|weight| *weight = (*weight + pair_score) / 2.0)
                    .or_insert(pair_score);
            }
        }
    }

    for ((left, right), weight) in anime_pair_weights {
        if let (Some(source), Some(target)) = (
            node_index.get(&format!("anime:{left}")),
            node_index.get(&format!("anime:{right}")),
        ) {
            let width = (0.35 + weight.abs() as f32 * 0.12).clamp(0.35, 2.2);
            edges.push(Edge {
                source: *source,
                target: *target,
                color: "#6fffe980",
                stroke_width: width,
            });
        }
    }

    layout_nodes(&mut nodes);

    let render_edges = edges
        .into_iter()
        .map(|edge| RenderEdge {
            x1: nodes[edge.source].x,
            y1: nodes[edge.source].y,
            x2: nodes[edge.target].x,
            y2: nodes[edge.target].y,
            color: edge.color,
            stroke_width: edge.stroke_width,
        })
        .collect::<Vec<_>>();

    let user_count = nodes.iter().filter(|n| n.node_type == NodeType::User).count();
    let anime_count = nodes.len() - user_count;

    GraphModel {
        user_count,
        anime_count,
        nodes,
        edges: render_edges,
    }
}

fn upsert_node(
    nodes: &mut Vec<Node>,
    node_index: &mut HashMap<String, usize>,
    id: String,
    label: String,
    node_type: NodeType,
) -> usize {
    if let Some(existing) = node_index.get(&id) {
        return *existing;
    }

    let node = match node_type {
        NodeType::User => Node {
            id: id.clone(),
            label,
            node_type,
            x: WIDTH / 2.0,
            y: HEIGHT / 2.0,
            radius: 7.0,
            color: "#ff8a00",
        },
        NodeType::Anime => Node {
            id: id.clone(),
            label,
            node_type,
            x: WIDTH / 2.0,
            y: HEIGHT / 2.0,
            radius: 3.8,
            color: "#0f8b8d",
        },
    };

    let idx = nodes.len();
    nodes.push(node);
    node_index.insert(id, idx);
    idx
}

fn layout_nodes(nodes: &mut [Node]) {
    let mut users = Vec::new();
    let mut anime = Vec::new();

    for (idx, node) in nodes.iter().enumerate() {
        if node.node_type == NodeType::User {
            users.push(idx);
        } else {
            anime.push(idx);
        }
    }

    for (i, idx) in users.iter().enumerate() {
        let angle = (i as f32 / users.len().max(1) as f32) * std::f32::consts::TAU;
        let radius = (HEIGHT.min(WIDTH) * 0.38).max(200.0);
        nodes[*idx].x = WIDTH / 2.0 + radius * angle.cos();
        nodes[*idx].y = HEIGHT / 2.0 + radius * angle.sin();
    }

    for (i, idx) in anime.iter().enumerate() {
        let angle = (i as f32 / anime.len().max(1) as f32) * std::f32::consts::TAU;
        let band = 120.0 + ((i % 7) as f32 * 17.0);
        let jitter = ((i * 29 % 17) as f32) - 8.0;
        nodes[*idx].x = WIDTH / 2.0 + (band + jitter) * angle.cos();
        nodes[*idx].y = HEIGHT / 2.0 + (band - jitter) * angle.sin();
    }
}

const APP_CSS: &str = r#"
  .app {
    margin: 0;
    min-height: 100vh;
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 16px;
    padding: 16px;
    background: radial-gradient(circle at 20% 20%, #2e5678 0%, transparent 45%),
      linear-gradient(160deg, #091019 0%, #17354f 100%);
    color: #f4f1de;
    font-family: Segoe UI, sans-serif;
    box-sizing: border-box;
  }
  .panel {
    border: 1px solid #ffffff26;
    border-radius: 14px;
    padding: 14px;
    background: #0e1723cc;
  }
  .muted {
    color: #b0b8c0;
    margin-top: 0;
  }
  .stats {
    margin-top: 14px;
    border: 1px solid #ffffff1f;
    border-radius: 12px;
    padding: 10px;
  }
  .row {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    padding: 2px 0;
  }
  .tiny {
    color: #b0b8c0;
    font-size: 12px;
  }
  .canvas-wrap {
    border: 1px solid #ffffff26;
    border-radius: 14px;
    overflow: hidden;
    background: #070d14;
  }
"#;

const SAMPLE_DATASET: &str = r#"
{
  "users": [
    {
      "userId": "desktopsample001",
      "ratings": [
        { "animeId": 1, "title": "Cowboy Bebop", "rawScore": 9, "normalizedScore": 0.0 },
        { "animeId": 1535, "title": "Death Note", "rawScore": 8, "normalizedScore": 0.0 },
        { "animeId": 16498, "title": "Shingeki no Kyojin", "rawScore": 7, "normalizedScore": 0.0 }
      ]
    },
    {
      "userId": "desktopsample002",
      "ratings": [
        { "animeId": 16498, "title": "Shingeki no Kyojin", "rawScore": 9, "normalizedScore": 0.0 },
        { "animeId": 30276, "title": "One Punch Man", "rawScore": 8, "normalizedScore": 0.0 },
        { "animeId": 11757, "title": "Sword Art Online", "rawScore": 6, "normalizedScore": 0.0 }
      ]
    }
  ]
}
"#;
