import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommend_graph_mf import Model, parse_watched_arg, recommend  # noqa: E402


class RecommendationSmokeTests(unittest.TestCase):
    def test_parse_watched_weights(self):
        self.assertEqual(parse_watched_arg("101, 102:1.5, 103:0.25"),
                         [(101, 1.0), (102, 1.5), (103, 0.25)])

    def test_synthetic_ranking_excludes_watched_item(self):
        model = Model(
            p=np.zeros((0, 2), dtype=np.float32),
            q=np.array([[1, 0], [0.8, 0], [-1, 0], [0, 0.2]], dtype=np.float32),
            bu=np.zeros(0, dtype=np.float32),
            bi=np.array([0, 0.1, 0, 0], dtype=np.float32),
            global_mean=0.0,
            user_ids=[],
            anime_ids=[101, 102, 103, 104],
            anime_titles=["Invented A", "Invented B", "Invented C", "Invented D"],
            train_user_items=[],
        )
        result = recommend(model, "", [(101, 1.0)], 2, 1, float("-inf"), {}, 0.0)
        ids = [item["animeId"] for item in result["recommendations"]]
        self.assertEqual(ids[0], 102)
        self.assertNotIn(101, ids)


if __name__ == "__main__":
    unittest.main()
