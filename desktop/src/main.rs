mod bundle;

use bundle::{load_bundle, load_demo_graph, CompactGraphV3, LoadedBundle};
use dioxus::prelude::*;
use rfd::FileDialog;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const WIDTH: f32 = 1040.0;
const HEIGHT: f32 = 760.0;
const MAX_RENDERED_NODES: usize = 300;
const MAX_RENDERED_EDGES: usize = 1400;

fn main() {
    dioxus::launch(App);
}

#[derive(Clone)]
struct DisplayNode {
    id: u64,
    x: f32,
    y: f32,
}

#[derive(Clone)]
struct DisplayEdge {
    left_id: u64,
    right_id: u64,
    left_title: String,
    right_title: String,
    weight: f64,
    support: usize,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    color: &'static str,
    stroke_width: f32,
}

#[derive(Clone)]
struct DisplayGraph {
    nodes: Vec<DisplayNode>,
    edges: Vec<DisplayEdge>,
}

#[derive(Clone)]
struct LoadedView {
    graph: CompactGraphV3,
    display: DisplayGraph,
    tag: Option<String>,
    bundle_id: Option<String>,
}

impl LoadedView {
    fn new(graph: CompactGraphV3, tag: Option<String>, bundle_id: Option<String>) -> Self {
        let display = build_display(&graph);
        Self {
            graph,
            display,
            tag,
            bundle_id,
        }
    }
}

#[derive(Clone)]
enum DataState {
    NoData,
    Loading,
    Demo(Rc<LoadedView>),
    Local(Rc<LoadedView>),
    Error(String),
}

fn selected_result(result: Result<LoadedBundle, String>) -> DataState {
    match result {
        Ok(bundle) => DataState::Local(Rc::new(LoadedView::new(
            bundle.graph,
            Some(bundle.tag),
            Some(bundle.bundle_id),
        ))),
        Err(error) => DataState::Error(error),
    }
}

#[component]
fn App() -> Element {
    let mut state = use_signal(|| DataState::NoData);
    let current = state.read().clone();
    let view = match &current {
        DataState::Demo(view) | DataState::Local(view) => Some(view.clone()),
        _ => None,
    };
    let status = match &current {
        DataState::NoData => "No data selected. Choose a local v3 release manifest or open the invented demo.".to_string(),
        DataState::Loading => "Loading selected local data…".to_string(),
        DataState::Demo(_) => "Invented demo. These titles and pairs are synthetic; no local file was selected.".to_string(),
        DataState::Local(_) => "Selected local data. The graph bytes and identity match the manifest; other assets and source permissions are not checked here.".to_string(),
        DataState::Error(error) => format!("Load failed. No graph is active. {error}"),
    };

    rsx! {
        style { {APP_CSS} }
        main { class: "app",
            section { class: "panel",
                h1 { "Anime graph companion" }
                p { class: "muted", "Inspect local, precomputed anime-pair evidence. This desktop app does not rank recommendations or import viewing history." }
                div { class: "actions",
                    button { onclick: move |_| {
                        if let Some(path) = FileDialog::new()
                            .set_title("Select release-manifest.json")
                            .add_filter("JSON manifest", &["json"])
                            .pick_file() {
                            state.set(DataState::Loading);
                            state.set(selected_result(load_bundle(&path)));
                        }
                    }, "Select local manifest" }
                    button { onclick: move |_| {
                        state.set(match load_demo_graph() {
                            Ok(graph) => DataState::Demo(Rc::new(LoadedView::new(graph, None, None))),
                            Err(error) => DataState::Error(error),
                        });
                    }, "Open invented demo" }
                    button { onclick: move |_| state.set(DataState::NoData), "Clear data" }
                }
                p { class: "status", role: "status", "{status}" }
                if let Some(view) = &view {
                    div { class: "stats",
                        StatRow { label: "Format", value: view.graph.format.clone() }
                        StatRow { label: "Role", value: view.graph.role.clone() }
                        StatRow { label: "Anime", value: view.graph.anime_count.to_string() }
                        StatRow { label: "Signed pairs", value: view.graph.edge_count.to_string() }
                        StatRow { label: "Selected ratings used for pairs", value: view.graph.truncation.selected_ratings.to_string() }
                        StatRow { label: "Omitted ratings", value: view.graph.truncation.ratings_skipped.to_string() }
                        StatRow { label: "Omitted pair visits", value: view.graph.truncation.pair_visits_skipped.to_string() }
                        StatRow { label: "Rendered titles", value: view.display.nodes.len().to_string() }
                        StatRow { label: "Rendered pairs", value: view.display.edges.len().to_string() }
                    }
                    p { class: "tiny", "Dataset source claim: {view.graph.dataset.source}" }
                    if let Some(tag) = &view.tag {
                        p { class: "tiny", "Selected manifest tag: {tag}" }
                    }
                    if let Some(bundle_id) = &view.bundle_id {
                        p { class: "tiny", "Declared bundle ID: {bundle_id}" }
                    }
                    p { class: "tiny", "The overview is limited to 300 connected titles and 1,400 pairs. Pair weights are signed mean centered preferences with co-rater support; they are not similarity scores. The graph keeps no user rows." }
                }
            }
            section { class: "canvas-wrap",
                if let Some(view) = &view {
                    svg {
                        width: "100%",
                        height: "100%",
                        view_box: "0 0 {WIDTH} {HEIGHT}",
                        role: "img",
                        "aria-label": "Bounded overview of precomputed signed anime pairs",
                        for edge in &view.display.edges {
                            line {
                                key: "{edge.left_id}-{edge.right_id}",
                                x1: "{edge.x1}", y1: "{edge.y1}",
                                x2: "{edge.x2}", y2: "{edge.y2}",
                                stroke: "{edge.color}", stroke_width: "{edge.stroke_width}",
                                stroke_opacity: "0.58"
                            }
                        }
                        for node in &view.display.nodes {
                            circle { key: "{node.id}", cx: "{node.x}", cy: "{node.y}", r: "5", fill: "#f4d35e" }
                        }
                    }
                    div { class: "evidence",
                        h2 { "Visible pair evidence" }
                        ul {
                            for edge in view.display.edges.iter().take(12) {
                                li { "{pair_label(edge)}" }
                            }
                        }
                    }
                } else {
                    p { class: "empty", "No graph to show." }
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

fn build_display(graph: &CompactGraphV3) -> DisplayGraph {
    let mut strength = vec![0.0; graph.anime.len()];
    for (left, right, weight, support) in &graph.aa {
        let evidence = weight.abs() * (*support as f64).ln_1p();
        strength[*left] += evidence;
        strength[*right] += evidence;
    }
    let mut selected: Vec<usize> = (0..graph.anime.len()).collect();
    selected.sort_by(|left, right| {
        strength[*right]
            .total_cmp(&strength[*left])
            .then_with(|| graph.anime[*left].0.cmp(&graph.anime[*right].0))
    });
    selected.truncate(MAX_RENDERED_NODES);
    selected.sort_by_key(|index| graph.anime[*index].0);

    let mut positions = HashMap::new();
    let nodes = selected
        .iter()
        .enumerate()
        .map(|(slot, index)| {
            let angle = (slot as f32 / selected.len().max(1) as f32) * std::f32::consts::TAU;
            let radius = 170.0 + (slot % 6) as f32 * 25.0;
            let x = WIDTH / 2.0 + radius * angle.cos();
            let y = HEIGHT / 2.0 + radius * angle.sin();
            positions.insert(*index, (x, y));
            DisplayNode {
                id: graph.anime[*index].0,
                x,
                y,
            }
        })
        .collect();

    let selected_set: HashSet<usize> = selected.into_iter().collect();
    let mut pairs: Vec<_> = graph
        .aa
        .iter()
        .filter(|(left, right, _, _)| selected_set.contains(left) && selected_set.contains(right))
        .collect();
    pairs.sort_by(|left, right| {
        right
            .2
            .abs()
            .total_cmp(&left.2.abs())
            .then_with(|| right.3.cmp(&left.3))
            .then_with(|| graph.anime[left.0].0.cmp(&graph.anime[right.0].0))
            .then_with(|| graph.anime[left.1].0.cmp(&graph.anime[right.1].0))
    });
    pairs.truncate(MAX_RENDERED_EDGES);
    let edges = pairs
        .into_iter()
        .map(|(left, right, weight, support)| {
            let (x1, y1) = positions[left];
            let (x2, y2) = positions[right];
            let color = if *weight > 0.0 {
                "#6fffe9"
            } else if *weight < 0.0 {
                "#ff8d80"
            } else {
                "#aab4c0"
            };
            DisplayEdge {
                left_id: graph.anime[*left].0,
                right_id: graph.anime[*right].0,
                left_title: graph.anime[*left].1.clone(),
                right_title: graph.anime[*right].1.clone(),
                weight: *weight,
                support: *support,
                x1,
                y1,
                x2,
                y2,
                color,
                stroke_width: (0.5 + weight.abs() as f32 * 0.16).clamp(0.5, 2.5),
            }
        })
        .collect();
    DisplayGraph { nodes, edges }
}

fn pair_label(edge: &DisplayEdge) -> String {
    format!(
        "{} ↔ {}: {:+.2} pair preference, {} co-rater{}",
        edge.left_title,
        edge.right_title,
        edge.weight,
        edge.support,
        if edge.support == 1 { "" } else { "s" }
    )
}

const APP_CSS: &str = r#"
  * { box-sizing: border-box; }
  body { margin: 0; }
  .app {
    min-height: 100vh; display: grid; grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
    gap: 16px; padding: 16px; background: linear-gradient(160deg, #091019 0%, #17354f 100%);
    color: #f4f1de; font-family: Segoe UI, sans-serif;
  }
  .panel { border: 1px solid #ffffff26; border-radius: 14px; padding: 16px; background: #0e1723cc; }
  .muted, .tiny { color: #b0b8c0; }
  .tiny { font-size: 12px; overflow-wrap: anywhere; }
  .actions { display: flex; flex-wrap: wrap; gap: 8px; }
  button { min-height: 40px; border: 1px solid #6fffe9; border-radius: 7px;
    background: #103345; color: #f4f1de; padding: 8px 12px; cursor: pointer; }
  button:focus-visible { outline: 2px solid #f4d35e; outline-offset: 2px; }
  .status { min-height: 3em; line-height: 1.4; }
  .stats { margin-top: 14px; border: 1px solid #ffffff1f; border-radius: 12px; padding: 10px; }
  .row { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; padding: 3px 0; }
  .row strong { text-align: right; }
  .canvas-wrap { border: 1px solid #ffffff26; border-radius: 14px; overflow: auto;
    background: #070d14; min-height: 600px; }
  .canvas-wrap svg { display: block; min-height: 520px; }
  .empty { padding: 24px; color: #b0b8c0; }
  .evidence { padding: 12px 20px 20px; }
  .evidence h2 { font-size: 17px; }
  .evidence li { margin: 4px 0; overflow-wrap: anywhere; }
  @media (max-width: 760px) { .app { grid-template-columns: 1fr; } }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_overview_uses_precomputed_signed_pairs_and_support() {
        let graph = load_demo_graph().unwrap();
        let display = build_display(&graph);
        assert_eq!(display.nodes.len(), graph.anime.len());
        assert_eq!(display.edges.len(), graph.aa.len());
        let expected: HashSet<_> = graph
            .aa
            .iter()
            .map(|(left, right, weight, support)| {
                (
                    graph.anime[*left].0,
                    graph.anime[*right].0,
                    weight.to_bits(),
                    *support,
                )
            })
            .collect();
        let actual: HashSet<_> = display
            .edges
            .iter()
            .map(|edge| {
                (
                    edge.left_id,
                    edge.right_id,
                    edge.weight.to_bits(),
                    edge.support,
                )
            })
            .collect();
        assert_eq!(actual, expected);
        assert!(display.edges.iter().any(|edge| edge.weight < 0.0));
        assert!(display.edges.iter().any(|edge| edge.weight > 0.0));
    }

    #[test]
    fn failed_selection_has_no_graph_and_demo_requires_a_separate_action() {
        assert!(matches!(
            selected_result(Err("invented error".into())),
            DataState::Error(_)
        ));
        assert!(matches!(DataState::NoData, DataState::NoData));
        assert!(matches!(
            DataState::Demo(Rc::new(LoadedView::new(
                load_demo_graph().unwrap(),
                None,
                None
            ))),
            DataState::Demo(_)
        ));
    }
}
