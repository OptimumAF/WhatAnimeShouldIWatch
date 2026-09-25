import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_graph_mf import load_anime_graph_edges  # noqa: E402


class SignedGraphEdgeTests(unittest.TestCase):
    def test_v1_negative_and_neutral_pairs_never_become_attractive_constraints(self):
        mapping = {101: 0, 102: 1, 103: 2}
        graphs = [
            {
                "format": "graph-compact-v1",
                "anime": [[101, "Invented A"], [102, "Invented B"], [103, "Invented C"]],
                "aa": [[0, 1, 0.8, 2], [0, 2, -0.7, 3], [1, 2, 0, 2]],
            },
            {
                "nodes": [],
                "edges": [
                    {"edgeType": "anime-anime", "source": "anime:101", "target": "anime:102", "weight": 0.8},
                    {"edgeType": "anime-anime", "source": "anime:101", "target": "anime:103", "weight": -0.7},
                    {"edgeType": "anime-anime", "source": "anime:102", "target": "anime:103", "weight": 0},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-graph.json"
            for graph in graphs:
                path.write_text(json.dumps(graph), encoding="utf-8")
                edges = load_anime_graph_edges(path, mapping, 0.0)
                self.assertEqual(edges.shape, (1, 3))
                self.assertEqual(edges[0, :2].tolist(), [0, 1])
                self.assertAlmostEqual(float(edges[0, 2]), 0.8)
                self.assertEqual(load_anime_graph_edges(path, mapping, 0.9).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
