"""Lightweight logic tests; run with: python -m unittest discover -s tests"""
import ast, math, pathlib, unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP = pathlib.Path(__file__).parents[1] / "app.py"
tree = ast.parse(APP.read_text())
selected = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"recommend_destinations", "haversine_km", "generate_itinerary"}:
        selected.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "INTEREST_TERMS" for t in node.targets):
        selected.append(node)
NS = {"pd": pd, "math": math, "TfidfVectorizer": TfidfVectorizer, "cosine_similarity": cosine_similarity}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(APP), "exec"), NS)

class WanderWiseLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(APP.parent / "data" / "indonesia_destinations.csv")

    def test_dataset_has_required_columns(self):
        required = {"city", "name", "category", "description", "rating", "estimated_cost_idr", "lat", "lon", "visit_minutes"}
        self.assertTrue(required.issubset(self.data.columns))

    def test_recommendations_are_ranked(self):
        bali = self.data[self.data.city == "Bali"]
        recs = NS["recommend_destinations"](bali, ["Beaches", "Nature"], 5_000_000, 3, 3.5)
        self.assertFalse(recs.empty)
        self.assertTrue(recs.score.is_monotonic_decreasing)
        self.assertTrue((recs.rating >= 3.5).all())

    def test_itinerary_generation(self):
        bali = self.data[self.data.city == "Bali"]
        recs = NS["recommend_destinations"](bali, ["Culture"], 5_000_000, 2, 3.0)
        plan = NS["generate_itinerary"](recs, 2, 2, 5_000_000)
        self.assertFalse(plan.empty)
        self.assertLessEqual(plan.Day.nunique(), 2)
        self.assertIn("Destination", plan.columns)

    def test_empty_recommendations_return_empty_plan(self):
        self.assertTrue(NS["generate_itinerary"](self.data.iloc[0:0], 2, 1, 100000).empty)

if __name__ == "__main__":
    unittest.main()
