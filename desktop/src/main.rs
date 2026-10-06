mod bundle;

use bundle::{load_bundle_with_cancel, load_demo_graph, CompactGraphV3, LoadedBundle};
use dioxus::prelude::*;
use rfd::AsyncFileDialog;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;

const WIDTH: f32 = 1040.0;
const HEIGHT: f32 = 760.0;
const MAX_RENDERED_NODES: usize = 300;
const MAX_RENDERED_EDGES: usize = 1400;
const MAX_VISIBLE_TITLE_CHARS: usize = 96;
const MAX_VISIBLE_SOURCE_CHARS: usize = 120;

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
    summary: ViewSummary,
    display: DisplayGraph,
    tag: Option<String>,
    bundle_id: Option<String>,
}

#[derive(Clone)]
struct ViewSummary {
    format: String,
    role: String,
    anime_count: usize,
    pair_count: usize,
    selected_ratings: usize,
    ratings_skipped: usize,
    pair_visits_skipped: usize,
    dataset_source: String,
}

impl LoadedView {
    fn new(graph: CompactGraphV3, tag: Option<String>, bundle_id: Option<String>) -> Self {
        let display = build_display(&graph);
        let summary = ViewSummary {
            format: graph.format,
            role: graph.role,
            anime_count: graph.anime_count,
            pair_count: graph.edge_count,
            selected_ratings: graph.truncation.selected_ratings,
            ratings_skipped: graph.truncation.ratings_skipped,
            pair_visits_skipped: graph.truncation.pair_visits_skipped,
            dataset_source: bounded_label(&graph.dataset.source, MAX_VISIBLE_SOURCE_CHARS),
        };
        Self {
            summary,
            display,
            tag: tag.map(|value| bounded_label(&value, MAX_VISIBLE_SOURCE_CHARS)),
            bundle_id,
        }
    }
}

#[derive(Clone)]
enum DataState {
    NoData,
    Choosing,
    Loading,
    Demo(Rc<LoadedView>),
    Local(Rc<LoadedView>),
    Error(String),
}

fn selected_result(result: Result<LoadedView, String>) -> DataState {
    match result {
        Ok(view) => DataState::Local(Rc::new(view)),
        Err(error) => DataState::Error(error),
    }
}

#[derive(Clone)]
struct LoadQueue {
    generation: Arc<AtomicU64>,
    slot: Arc<Semaphore>,
}

impl LoadQueue {
    fn new() -> Self {
        Self {
            generation: Arc::new(AtomicU64::new(0)),
            slot: Arc::new(Semaphore::new(1)),
        }
    }

    fn next(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn is_current(&self, request: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == request
    }

    async fn load_path(
        &self,
        path: std::path::PathBuf,
        request: u64,
    ) -> Option<Result<LoadedView, String>> {
        self.run_with(request, move |cancelled| {
            load_bundle_with_cancel(&path, cancelled).map(|loaded| {
                loaded.map(
                    |LoadedBundle {
                         tag,
                         bundle_id,
                         graph,
                     }| { LoadedView::new(graph, Some(tag), Some(bundle_id)) },
                )
            })
        })
        .await
    }

    async fn run_with<F>(&self, request: u64, loader: F) -> Option<Result<LoadedView, String>>
    where
        F: FnOnce(&dyn Fn() -> bool) -> Result<Option<LoadedView>, String> + Send + 'static,
    {
        let permit = match self.slot.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => return Some(Err("Local data loader is unavailable".into())),
        };
        if !self.is_current(request) {
            return None;
        }
        let generation = self.generation.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let cancelled = || generation.load(Ordering::SeqCst) != request;
            if cancelled() {
                return None;
            }
            let loaded = loader(&cancelled);
            if cancelled() {
                return None;
            }
            match loaded {
                Ok(Some(view)) => Some(Ok(view)),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .await;
        if !self.is_current(request) {
            return None;
        }
        match result {
            Ok(value) => value,
            Err(_) => Some(Err("Local background load failed".into())),
        }
    }
}

#[component]
fn App() -> Element {
    let mut state = use_signal(|| DataState::NoData);
    let queue = use_hook(LoadQueue::new);
    let select_queue = queue.clone();
    let demo_queue = queue.clone();
    let clear_queue = queue.clone();
    let current = state.read().clone();
    let view = match &current {
        DataState::Demo(view) | DataState::Local(view) => Some(view.clone()),
        _ => None,
    };
    let status = match &current {
        DataState::NoData => "No data selected. Choose a local v3 release manifest or open the invented demo.".to_string(),
        DataState::Choosing => "Choose a local release manifest. Cancel to return to the previous view.".to_string(),
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
                        let request = select_queue.next();
                        let previous = state.read().clone();
                        state.set(DataState::Choosing);
                        let queue = select_queue.clone();
                        spawn(async move {
                            let picked = AsyncFileDialog::new()
                                .set_title("Select release-manifest.json")
                                .add_filter("JSON manifest", &["json"])
                                .pick_file()
                                .await;
                            if !queue.is_current(request) {
                                return;
                            }
                            if let Some(file) = picked {
                                state.set(DataState::Loading);
                                if let Some(result) = queue.load_path(file.path().to_path_buf(), request).await {
                                    if queue.is_current(request) {
                                        state.set(selected_result(result));
                                    }
                                }
                            } else {
                                state.set(if matches!(previous, DataState::Loading | DataState::Choosing) {
                                    DataState::NoData
                                } else {
                                    previous
                                });
                            }
                        });
                    }, "Select local manifest" }
                    button { onclick: move |_| {
                        demo_queue.next();
                        state.set(match load_demo_graph() {
                            Ok(graph) => DataState::Demo(Rc::new(LoadedView::new(graph, None, None))),
                            Err(error) => DataState::Error(error),
                        });
                    }, "Open invented demo" }
                    button { onclick: move |_| {
                        clear_queue.next();
                        state.set(DataState::NoData);
                    }, "Clear data" }
                }
                p { class: "status", role: "status", "{status}" }
                if let Some(view) = &view {
                    div { class: "stats",
                        StatRow { label: "Format", value: view.summary.format.clone() }
                        StatRow { label: "Role", value: view.summary.role.clone() }
                        StatRow { label: "Anime", value: view.summary.anime_count.to_string() }
                        StatRow { label: "Signed pairs", value: view.summary.pair_count.to_string() }
                        StatRow { label: "Selected ratings used for pairs", value: view.summary.selected_ratings.to_string() }
                        StatRow { label: "Omitted ratings", value: view.summary.ratings_skipped.to_string() }
                        StatRow { label: "Omitted pair visits", value: view.summary.pair_visits_skipped.to_string() }
                        StatRow { label: "Rendered titles", value: view.display.nodes.len().to_string() }
                        StatRow { label: "Rendered pairs", value: view.display.edges.len().to_string() }
                    }
                    p { class: "tiny", "Dataset source claim: {view.summary.dataset_source}" }
                    if let Some(tag) = &view.tag {
                        p { class: "tiny", "Selected manifest tag (shortened if long): {tag}" }
                    }
                    if let Some(bundle_id) = &view.bundle_id {
                        p { class: "tiny", "Declared bundle ID: {bundle_id}" }
                    }
                    p { class: "tiny", "The overview is limited to 300 connected titles and 1,400 pairs; long labels are shortened. Pair weights are signed mean centered preferences with co-rater support; they are not similarity scores. The graph keeps no user rows." }
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
    // Keep only the strongest pair indexes while scanning the full graph. This
    // avoids a large sort and prevents scattered high-strength nodes from
    // producing an almost empty overview on a sparse graph.
    let mut seed_pairs = BinaryHeap::new();
    for index in 0..graph.aa.len() {
        keep_best_pair(
            &mut seed_pairs,
            pair_priority(graph, index),
            MAX_RENDERED_EDGES * 4,
        );
    }
    let mut ranked_seed_pairs: Vec<_> = seed_pairs.into_iter().map(|Reverse(key)| key).collect();
    ranked_seed_pairs.sort_unstable_by(|left, right| right.cmp(left));
    let mut selected_set = HashSet::new();
    for key in ranked_seed_pairs {
        let (_, _, _, _, Reverse(index)) = key;
        let (left, right, _, _) = graph.aa[index];
        let needed = usize::from(!selected_set.contains(&left))
            + usize::from(!selected_set.contains(&right));
        if selected_set.len() + needed <= MAX_RENDERED_NODES {
            selected_set.insert(left);
            selected_set.insert(right);
        }
    }
    if selected_set.len() < MAX_RENDERED_NODES {
        for index in 0..graph.anime.len() {
            selected_set.insert(index);
            if selected_set.len() == MAX_RENDERED_NODES {
                break;
            }
        }
    }
    let mut selected: Vec<_> = selected_set.iter().copied().collect();
    selected.sort_unstable_by_key(|index| graph.anime[*index].0);

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

    let mut visible_pairs = BinaryHeap::new();
    for (index, (left, right, _, _)) in graph.aa.iter().enumerate() {
        if selected_set.contains(left) && selected_set.contains(right) {
            keep_best_pair(
                &mut visible_pairs,
                pair_priority(graph, index),
                MAX_RENDERED_EDGES,
            );
        }
    }
    let mut ranked_pairs: Vec<_> = visible_pairs.into_iter().map(|Reverse(key)| key).collect();
    ranked_pairs.sort_unstable_by(|left, right| right.cmp(left));
    let edges = ranked_pairs
        .into_iter()
        .map(|key| {
            let (_, _, _, _, Reverse(index)) = key;
            let (left, right, weight, support) = graph.aa[index];
            let (x1, y1) = positions[&left];
            let (x2, y2) = positions[&right];
            let color = if weight > 0.0 {
                "#6fffe9"
            } else if weight < 0.0 {
                "#ff8d80"
            } else {
                "#aab4c0"
            };
            DisplayEdge {
                left_id: graph.anime[left].0,
                right_id: graph.anime[right].0,
                left_title: bounded_label(&graph.anime[left].1, MAX_VISIBLE_TITLE_CHARS),
                right_title: bounded_label(&graph.anime[right].1, MAX_VISIBLE_TITLE_CHARS),
                weight,
                support,
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

// Greater keys mean stronger evidence; the reverse heap keeps its weakest
// retained pair at the top for bounded replacement.
type PairPriority = (u64, usize, Reverse<u64>, Reverse<u64>, Reverse<usize>);

fn pair_priority(graph: &CompactGraphV3, index: usize) -> PairPriority {
    let (left, right, weight, support) = graph.aa[index];
    let evidence = weight.abs() * (support as f64).ln_1p();
    let left_id = graph.anime[left].0;
    let right_id = graph.anime[right].0;
    (
        evidence.to_bits(),
        support,
        Reverse(left_id.min(right_id)),
        Reverse(left_id.max(right_id)),
        Reverse(index),
    )
}

fn keep_best_pair(heap: &mut BinaryHeap<Reverse<PairPriority>>, key: PairPriority, limit: usize) {
    if heap.len() < limit {
        heap.push(Reverse(key));
    } else if heap.peek().is_some_and(|weakest| key > weakest.0) {
        heap.pop();
        heap.push(Reverse(key));
    }
}

fn bounded_label(value: &str, limit: usize) -> String {
    let mut characters = value.chars();
    let visible: String = characters.by_ref().take(limit).collect();
    if characters.next().is_some() {
        format!("{visible}…")
    } else {
        visible
    }
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
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::time::Duration;

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
    fn failed_selection_has_no_graph() {
        match selected_result(Err("invented error".into())) {
            DataState::Error(message) => assert_eq!(message, "invented error"),
            _ => panic!("a failed local selection must not show a graph"),
        }
    }

    #[test]
    fn large_unicode_graph_keeps_display_bounded_and_rejects_a_late_bad_pair() {
        const ANIME: usize = 5_000;
        const PAIRS: usize = 20_000;
        let mut value: Value =
            serde_json::from_slice(include_bytes!("../fixtures/graph.compact.json")).unwrap();
        value["anime"] = Value::Array(
            (0..ANIME)
                .map(|index| {
                    json!([
                        index + 1,
                        format!("星の航路 {index} 🌌{}", "界".repeat(200))
                    ])
                })
                .collect(),
        );
        value["aa"] = Value::Array(
            (0..PAIRS)
                .map(|index| {
                    let left = index % ANIME;
                    let right = (left + 1 + index / ANIME) % ANIME;
                    let weight = if index % 3 == 0 { -0.75 } else { 0.5 };
                    json!([left, right, weight, 1 + index % 5])
                })
                .collect(),
        );
        value["animeCount"] = json!(ANIME);
        value["nodeCount"] = json!(ANIME);
        value["edgeCount"] = json!(PAIRS);
        value["truncation"]["inputRatings"] = json!(ANIME * 10);
        value["truncation"]["selectedRatings"] = json!(ANIME * 10);
        value["truncation"]["potentialPairVisits"] = json!(PAIRS * 2);
        value["truncation"]["pairVisits"] = json!(PAIRS * 2);
        value["truncation"]["candidatePairs"] = json!(PAIRS);
        value["truncation"]["eligiblePairs"] = json!(PAIRS);
        value["truncation"]["selectedPairs"] = json!(PAIRS);
        // This exercises Rust parsing/display bounds; it is not a complete release bundle.
        let bytes = serde_json::to_vec(&value).unwrap();
        let graph = bundle::parse_graph(&bytes).unwrap();
        let source: HashSet<_> = graph
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
        let view = LoadedView::new(graph, None, None);
        assert_eq!(view.summary.anime_count, ANIME);
        assert_eq!(view.summary.pair_count, PAIRS);
        assert_eq!(view.display.nodes.len(), MAX_RENDERED_NODES);
        assert!(view.display.edges.len() >= 100);
        assert!(view.display.edges.len() <= MAX_RENDERED_EDGES);
        assert!(view
            .display
            .nodes
            .iter()
            .all(|node| node.id <= ANIME as u64));
        for edge in &view.display.edges {
            assert!(edge.left_title.contains('星'));
            assert!(edge.left_title.chars().count() <= MAX_VISIBLE_TITLE_CHARS + 1);
            assert!(edge.left_title.ends_with('…'));
            assert!(edge.right_title.chars().count() <= MAX_VISIBLE_TITLE_CHARS + 1);
            assert!(source.contains(&(
                edge.left_id,
                edge.right_id,
                edge.weight.to_bits(),
                edge.support,
            )));
        }
        value["aa"][PAIRS - 1][3] = json!(0);
        assert!(bundle::parse_graph(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .contains(&format!("aa[{}]", PAIRS - 1)));
    }

    #[test]
    fn obsolete_loads_cannot_replace_the_latest_view_or_run_in_parallel() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let queue = LoadQueue::new();
            let first = queue.next();
            let (started_tx, started_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let first_queue = queue.clone();
            let first_task = tokio::spawn(async move {
                first_queue
                    .run_with(first, move |_| {
                        started_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        Ok(Some(LoadedView::new(
                            load_demo_graph().unwrap(),
                            None,
                            None,
                        )))
                    })
                    .await
            });
            tokio::task::yield_now().await;
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let second = queue.next();
            let started_second = Arc::new(AtomicUsize::new(0));
            let second_count = started_second.clone();
            let second_queue = queue.clone();
            let second_task = tokio::spawn(async move {
                second_queue
                    .run_with(second, move |_| {
                        second_count.fetch_add(1, Ordering::SeqCst);
                        Ok(Some(LoadedView::new(
                            load_demo_graph().unwrap(),
                            None,
                            None,
                        )))
                    })
                    .await
            });
            tokio::task::yield_now().await;
            assert_eq!(started_second.load(Ordering::SeqCst), 0);
            let latest = queue.next();
            release_tx.send(()).unwrap();
            assert!(first_task.await.unwrap().is_none());
            assert!(second_task.await.unwrap().is_none());
            assert_eq!(started_second.load(Ordering::SeqCst), 0);
            let current = queue
                .run_with(latest, |_| {
                    Ok(Some(LoadedView::new(
                        load_demo_graph().unwrap(),
                        None,
                        None,
                    )))
                })
                .await;
            assert!(matches!(current, Some(Ok(_))));
        });
    }
}
