use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const MANIFEST_LIMIT: u64 = 256 * 1024;
const GRAPH_LIMIT: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DatasetIdentity {
    pub sha256: String,
    pub scope: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Truncation {
    pub input_ratings: usize,
    pub selected_ratings: usize,
    pub ratings_skipped: usize,
    pub potential_pair_visits: usize,
    pub pair_visits: usize,
    pub pair_visits_skipped: usize,
    pub candidate_pairs: usize,
    pub eligible_pairs: usize,
    pub selected_pairs: usize,
    pub excluded_by_support: usize,
    pub excluded_by_neighbor_limit: usize,
    pub excluded_by_output_limit: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Semantics {
    pair_weight: String,
    support: String,
    recommendation_use: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GraphConfig {
    rating_selection_policy: String,
    seed: u32,
    max_ratings_per_user: usize,
    max_anime_anime_edges: usize,
    max_pair_visits: usize,
    max_pair_candidates: usize,
    min_pair_support: usize,
    max_neighbors_per_anime: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Projection {
    policy: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CompactGraphV3 {
    pub format: String,
    pub role: String,
    pub graph_id: String,
    pub dataset: DatasetIdentity,
    semantics: Semantics,
    config: GraphConfig,
    pub truncation: Truncation,
    projection: Projection,
    generated_at: String,
    user_ids: Vec<serde_json::Value>,
    pub anime: Vec<(u64, String)>,
    ua: Vec<serde_json::Value>,
    pub aa: Vec<(usize, usize, f64, usize)>,
    user_count: usize,
    pub anime_count: usize,
    node_count: usize,
    pub edge_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogAsset {
    path: String,
    format: String,
    sha256: String,
    bytes: u64,
    anime_count: usize,
    item_map_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GraphAsset {
    path: String,
    format: String,
    sha256: String,
    bytes: u64,
    graph_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExplorerAsset {
    path: String,
    format: String,
    sha256: String,
    bytes: u64,
    graph_id: String,
    source_graph_id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Coverage {
    mapped_anime_count: usize,
    total_catalog_anime_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelAsset {
    path: String,
    format: String,
    sha256: String,
    bytes: u64,
    dataset_sha256: String,
    item_map_sha256: String,
    coverage: Coverage,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PreviousBundle {
    tag: String,
    bundle_id: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    format: String,
    tag: String,
    bundle_id: String,
    dataset: DatasetIdentity,
    catalog: CatalogAsset,
    neighborhood: GraphAsset,
    explorer: ExplorerAsset,
    model: Option<ModelAsset>,
    last_known_good: Option<PreviousBundle>,
}

#[derive(Clone, Debug)]
pub struct LoadedBundle {
    pub tag: String,
    pub bundle_id: String,
    pub graph: CompactGraphV3,
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase())
}

fn require_digest(value: &str, field: &str) -> Result<(), String> {
    if digest(value) {
        Ok(())
    } else {
        Err(format!(
            "release-manifest.json: {field} must be a lowercase SHA-256 digest"
        ))
    }
}

fn require_asset(
    path: &str,
    format: &str,
    sha256: &str,
    bytes: u64,
    field: &str,
    expected_path: &str,
    expected_format: &str,
) -> Result<(), String> {
    if path != expected_path {
        return Err(format!(
            "release-manifest.json: {field}.path must be {expected_path}"
        ));
    }
    if format != expected_format {
        return Err(format!(
            "release-manifest.json: {field}.format must be {expected_format}"
        ));
    }
    require_digest(sha256, &format!("{field}.sha256"))?;
    if bytes == 0 || bytes > GRAPH_LIMIT {
        return Err(format!(
            "release-manifest.json: {field}.bytes is outside the supported bound"
        ));
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.format != "release-manifest-v1" {
        return Err("release-manifest.json: format must be release-manifest-v1".into());
    }
    if !manifest.tag.starts_with("data-v")
        || manifest.tag.len() <= 6
        || !manifest.tag[6..]
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || b"._-".contains(&ch))
    {
        return Err("release-manifest.json: tag must be a versioned data-v tag".into());
    }
    require_digest(&manifest.bundle_id, "bundleId")?;
    require_digest(&manifest.dataset.sha256, "dataset.sha256")?;
    if manifest.dataset.scope != "anonymized-ratings-content-v1"
        || manifest.dataset.source.trim().is_empty()
    {
        return Err("release-manifest.json: dataset scope/source is unsupported".into());
    }
    require_asset(
        &manifest.catalog.path,
        &manifest.catalog.format,
        &manifest.catalog.sha256,
        manifest.catalog.bytes,
        "catalog",
        "catalog.identity.json",
        "anime-catalog-v1",
    )?;
    if manifest.catalog.anime_count == 0 {
        return Err("release-manifest.json: catalog.animeCount must be positive".into());
    }
    require_digest(&manifest.catalog.item_map_sha256, "catalog.itemMapSha256")?;
    require_asset(
        &manifest.neighborhood.path,
        &manifest.neighborhood.format,
        &manifest.neighborhood.sha256,
        manifest.neighborhood.bytes,
        "neighborhood",
        "graph.compact.json",
        "graph-compact-v3",
    )?;
    require_digest(&manifest.neighborhood.graph_id, "neighborhood.graphId")?;
    require_asset(
        &manifest.explorer.path,
        &manifest.explorer.format,
        &manifest.explorer.sha256,
        manifest.explorer.bytes,
        "explorer",
        "graph-explorer.compact.json",
        "graph-compact-v3",
    )?;
    require_digest(&manifest.explorer.graph_id, "explorer.graphId")?;
    if manifest.explorer.source_graph_id != manifest.neighborhood.graph_id {
        return Err(
            "release-manifest.json: explorer.sourceGraphId differs from neighborhood.graphId"
                .into(),
        );
    }
    if let Some(model) = &manifest.model {
        require_asset(
            &model.path,
            &model.format,
            &model.sha256,
            model.bytes,
            "model",
            "model-mf-web.compact.json",
            "model-mf-compact-v1",
        )?;
        require_digest(&model.dataset_sha256, "model.datasetSha256")?;
        require_digest(&model.item_map_sha256, "model.itemMapSha256")?;
        if model.dataset_sha256 != manifest.dataset.sha256
            || model.coverage.mapped_anime_count == 0
            || model.coverage.mapped_anime_count > manifest.catalog.anime_count
            || model.coverage.total_catalog_anime_count != manifest.catalog.anime_count
        {
            return Err(
                "release-manifest.json: model dataset/coverage differs from catalog".into(),
            );
        }
    }
    if let Some(previous) = &manifest.last_known_good {
        if previous.tag == manifest.tag {
            return Err("release-manifest.json: lastKnownGood.tag must differ from tag".into());
        }
        require_digest(&previous.bundle_id, "lastKnownGood.bundleId")?;
        require_digest(&previous.manifest_sha256, "lastKnownGood.manifestSha256")?;
    }
    Ok(())
}

fn checked_sum(left: usize, right: usize, field: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("graph.compact.json: {field} overflows"))
}

pub fn parse_graph(bytes: &[u8]) -> Result<CompactGraphV3, String> {
    let graph: CompactGraphV3 =
        serde_json::from_slice(bytes).map_err(|error| format!("graph.compact.json: {error}"))?;
    if graph.format != "graph-compact-v3" || graph.role != "recommendation" {
        return Err("graph.compact.json: format/role must be v3 recommendation".into());
    }
    if !digest(&graph.graph_id)
        || !digest(&graph.dataset.sha256)
        || graph.dataset.scope != "anonymized-ratings-content-v1"
        || graph.dataset.source.trim().is_empty()
    {
        return Err("graph.compact.json: graphId/dataset is invalid".into());
    }
    if graph.semantics.pair_weight != "centered-pair-preference-mean-v1"
        || graph.semantics.support != "co-raters-after-selection-v1"
        || graph.semantics.recommendation_use != "positive-only-v1"
    {
        return Err("graph.compact.json: semantics is unsupported".into());
    }
    if graph.projection.policy != "omit-user-anime-v1"
        || !graph.user_ids.is_empty()
        || !graph.ua.is_empty()
        || graph.user_count != 0
    {
        return Err("graph.compact.json: projection/userIds/ua must be aggregate-only".into());
    }
    if graph.config.rating_selection_policy
        != if graph.config.max_ratings_per_user > 0 {
            "sha256-bottom-k-v1"
        } else {
            "all-ratings"
        }
        || graph.config.max_pair_visits == 0
        || graph.config.max_pair_candidates == 0
        || graph.config.min_pair_support == 0
    {
        return Err("graph.compact.json: config is invalid".into());
    }
    // These declared limits are retained as evidence even when the view renders a subset.
    let _ = (
        graph.config.seed,
        graph.config.max_anime_anime_edges,
        graph.config.max_neighbors_per_anime,
    );
    if graph.generated_at.trim().is_empty() {
        return Err("graph.compact.json: generatedAt is required".into());
    }
    let truncation = &graph.truncation;
    if checked_sum(
        truncation.selected_ratings,
        truncation.ratings_skipped,
        "truncation.inputRatings",
    )? != truncation.input_ratings
        || checked_sum(
            truncation.pair_visits,
            truncation.pair_visits_skipped,
            "truncation.potentialPairVisits",
        )? != truncation.potential_pair_visits
        || checked_sum(
            truncation.eligible_pairs,
            truncation.excluded_by_support,
            "truncation.candidatePairs",
        )? != truncation.candidate_pairs
        || checked_sum(
            checked_sum(
                truncation.selected_pairs,
                truncation.excluded_by_neighbor_limit,
                "truncation.eligiblePairs",
            )?,
            truncation.excluded_by_output_limit,
            "truncation.eligiblePairs",
        )? != truncation.eligible_pairs
    {
        return Err("graph.compact.json: truncation counts do not reconcile".into());
    }
    if graph.anime_count == 0
        || graph.anime_count != graph.anime.len()
        || graph.node_count != graph.anime.len()
        || graph.edge_count != graph.aa.len()
        || truncation.selected_pairs != graph.aa.len()
    {
        return Err(
            "graph.compact.json: animeCount/nodeCount/edgeCount/selectedPairs differs from arrays"
                .into(),
        );
    }
    let mut anime_ids = HashSet::new();
    for (index, (id, title)) in graph.anime.iter().enumerate() {
        if *id == 0
            || *id > 9_007_199_254_740_991
            || title.trim().is_empty()
            || !anime_ids.insert(*id)
        {
            return Err(format!(
                "graph.compact.json: anime[{index}] has an invalid ID/title"
            ));
        }
    }
    let mut pairs = HashSet::new();
    for (index, (left, right, weight, support)) in graph.aa.iter().enumerate() {
        if *left >= graph.anime.len()
            || *right >= graph.anime.len()
            || left == right
            || !weight.is_finite()
            || *support == 0
            || !pairs.insert(((*left).min(*right), (*left).max(*right)))
        {
            return Err(format!(
                "graph.compact.json: aa[{index}] is invalid or duplicated"
            ));
        }
    }
    Ok(graph)
}

fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    let length = fs::metadata(path)
        .map_err(|_| format!("{label}: missing or unreadable"))?
        .len();
    if length == 0 || length > limit {
        return Err(format!(
            "{label}: byte length is outside the supported bound"
        ));
    }
    let bytes = fs::read(path).map_err(|_| format!("{label}: missing or unreadable"))?;
    if bytes.len() as u64 != length || bytes.len() as u64 > limit {
        return Err(format!("{label}: byte length changed during read"));
    }
    Ok(bytes)
}

pub fn load_bundle(manifest_path: &Path) -> Result<LoadedBundle, String> {
    let manifest_bytes = read_bounded(manifest_path, MANIFEST_LIMIT, "release-manifest.json")?;
    let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("release-manifest.json: {error}"))?;
    let object = manifest_value
        .as_object()
        .ok_or("release-manifest.json: root must be an object")?;
    if !object.contains_key("model") || !object.contains_key("lastKnownGood") {
        return Err("release-manifest.json: model and lastKnownGood fields are required".into());
    }
    let manifest: Manifest = serde_json::from_value(manifest_value)
        .map_err(|error| format!("release-manifest.json: {error}"))?;
    validate_manifest(&manifest)?;
    let directory = manifest_path
        .parent()
        .ok_or("release-manifest.json: no containing directory")?;
    let graph_bytes = read_bounded(
        &directory.join("graph.compact.json"),
        GRAPH_LIMIT,
        "graph.compact.json",
    )?;
    if graph_bytes.len() as u64 != manifest.neighborhood.bytes
        || format!("{:x}", Sha256::digest(&graph_bytes)) != manifest.neighborhood.sha256
    {
        return Err("graph.compact.json: byte length or SHA-256 differs from release-manifest.json.neighborhood".into());
    }
    let graph = parse_graph(&graph_bytes)?;
    if graph.graph_id != manifest.neighborhood.graph_id
        || graph.dataset != manifest.dataset
        || graph.anime_count != manifest.catalog.anime_count
    {
        return Err(
            "graph.compact.json: graphId/dataset/animeCount differs from release-manifest.json"
                .into(),
        );
    }
    Ok(LoadedBundle {
        tag: manifest.tag,
        bundle_id: manifest.bundle_id,
        graph,
    })
}

pub fn load_demo_graph() -> Result<CompactGraphV3, String> {
    parse_graph(include_bytes!("../fixtures/graph.compact.json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn fixture_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/release-manifest.json")
    }

    fn copy_fixture() -> TempDir {
        let directory = TempDir::new().unwrap();
        let source = fixture_path().parent().unwrap().to_path_buf();
        for file in ["release-manifest.json", "graph.compact.json"] {
            fs::copy(source.join(file), directory.path().join(file)).unwrap();
        }
        directory
    }

    #[test]
    fn invented_v3_bundle_loads_with_manifest_identity() {
        let bundle = load_bundle(&fixture_path()).unwrap();
        assert_eq!(bundle.tag, "data-vsynthetic-desktop-v1");
        assert_eq!(bundle.graph.anime_count, 8);
        assert_eq!(bundle.graph.aa.len(), 11);
        assert_eq!(bundle.graph.dataset.source, "synthetic-fixture");
        assert_eq!(bundle.graph.user_count, 0);
        assert!(digest(&bundle.bundle_id));
    }

    #[test]
    fn a_selected_graph_cannot_fall_back_after_byte_or_schema_failure() {
        let directory = copy_fixture();
        let graph_path = directory.path().join("graph.compact.json");
        fs::write(&graph_path, b"{}").unwrap();
        let error = load_bundle(&directory.path().join("release-manifest.json")).unwrap_err();
        assert!(error.contains("graph.compact.json: byte length or SHA-256"));
        fs::remove_file(graph_path).unwrap();
        let error = load_bundle(&directory.path().join("release-manifest.json")).unwrap_err();
        assert!(error.contains("graph.compact.json: missing or unreadable"));
    }

    #[test]
    fn rejects_legacy_and_user_linked_graphs_and_preserves_unicode() {
        let source: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/graph.compact.json")).unwrap();
        assert!(
            parse_graph(include_bytes!("../fixtures/graph.compact.json"))
                .unwrap()
                .anime
                .iter()
                .any(|(_, title)| !title.is_ascii())
        );
        let mut legacy = source.clone();
        legacy["format"] = json!("graph-compact-v2");
        assert!(parse_graph(&serde_json::to_vec(&legacy).unwrap())
            .unwrap_err()
            .contains("format/role"));
        let mut linked = source.clone();
        linked["userIds"] = json!(["invented-user"]);
        assert!(parse_graph(&serde_json::to_vec(&linked).unwrap())
            .unwrap_err()
            .contains("projection/userIds/ua"));
        let mut hidden = source;
        hidden["privateRatings"] = json!([1]);
        assert!(parse_graph(&serde_json::to_vec(&hidden).unwrap())
            .unwrap_err()
            .contains("unknown field"));
    }

    #[test]
    fn rejects_bad_pairs_and_manifest_fields() {
        let mut graph: serde_json::Value =
            serde_json::from_slice(include_bytes!("../fixtures/graph.compact.json")).unwrap();
        graph["aa"][0][3] = json!(0);
        assert!(parse_graph(&serde_json::to_vec(&graph).unwrap())
            .unwrap_err()
            .contains("aa[0]"));
        let directory = copy_fixture();
        let manifest_path = directory.path().join("release-manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["neighborhood"]["format"] = json!("graph-compact-v2");
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(load_bundle(&manifest_path)
            .unwrap_err()
            .contains("neighborhood.format"));
    }
}
