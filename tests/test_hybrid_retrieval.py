import unittest

from app.services.retrieval import _hybrid_score_details


class HybridRetrievalTests(unittest.TestCase):
    def test_hybrid_score_details_include_semantic_lexical_and_hybrid_values(self):
        query = "What is LOP?"
        texts = ["LOP means line of progress for employees", "This is unrelated text"]
        details = _hybrid_score_details(query, texts)

        self.assertEqual(len(details), 2)
        for item in details:
            self.assertIn("semantic_score", item)
            self.assertIn("lexical_score", item)
            self.assertIn("hybrid_score", item)
            self.assertGreaterEqual(item["hybrid_score"], 0.0)
            self.assertLessEqual(item["hybrid_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
