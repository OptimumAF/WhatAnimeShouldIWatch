import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_graph_mf import load_anime_graph_edges, verify_graph_dataset_identity  # noqa: E402


class GraphV2ContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture_dir = Path(__file__).resolve().parents[2] / "web" / "public" / "demo-data"
        self.graph = json.loads((self.fixture_dir / "graph.compact.json").read_text(encoding="utf-8"))
        self.explorer = json.loads((self.fixture_dir / "graph-explorer.compact.json").read_text(encoding="utf-8"))
        self.mapping = {anime_id: index for index, (anime_id, _) in enumerate(self.graph["anime"])}

    def test_v2_trainer_accepts_recommendation_role_and_rejects_visualization_or_semantic_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-graph.json"
            path.write_text(json.dumps(self.graph), encoding="utf-8")
            edges = load_anime_graph_edges(path, self.mapping, 0.0)
            self.assertGreater(edges.shape[0], 0)
            path.write_text(json.dumps(self.explorer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "role must be recommendation"):
                load_anime_graph_edges(path, self.mapping, 0.0)
            changed = json.loads(json.dumps(self.graph))
            changed["semantics"]["pairWeight"] = "invented-other-statistic"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported v2 graph semantics"):
                load_anime_graph_edges(path, self.mapping, 0.0)
            changed["format"] = "graph-compact-v3"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported graph format"):
                load_anime_graph_edges(path, self.mapping, 0.0)
            changed = json.loads(json.dumps(self.graph))
            changed["format"] = "graph-compact-v1"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata requires a v2"):
                load_anime_graph_edges(path, self.mapping, 0.0)
            for change, expected in (
                (lambda item: item["aa"][0].pop(), "pair support"),
                (lambda item: item["aa"][0].__setitem__(1, 9999), "invalid reference"),
                (lambda item: item["config"].__setitem__("seed", -1), "config.seed"),
                (lambda item: item["truncation"].__setitem__("selectedPairs", 0), "do not reconcile"),
            ):
                changed = json.loads(json.dumps(self.graph))
                change(changed)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    load_anime_graph_edges(path, self.mapping, 0.0)

    def test_v2_training_checks_ratings_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.json"
            ratings_path = Path(directory) / "ratings.json"
            graph_path.write_text(json.dumps(self.graph), encoding="utf-8")
            ratings_path.write_text(json.dumps({"datasetSha256": self.graph["dataset"]["sha256"]}), encoding="utf-8")
            verify_graph_dataset_identity(graph_path, ratings_path)
            ratings_path.write_text(json.dumps({"datasetSha256": "0" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity does not match"):
                verify_graph_dataset_identity(graph_path, ratings_path)


if __name__ == "__main__":
    unittest.main()
