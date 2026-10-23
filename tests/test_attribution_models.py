import unittest
from unittest.mock import patch

import numpy as np

from attribution import _alignment_scorer, get_sentence_attribution
from schemas import QARequest, SummarizeRequest


class AttributionModelTests(unittest.TestCase):
    def test_text_requests_default_to_pretrained_sst2_only(self):
        self.assertEqual(SummarizeRequest(text="A passage.").attribution_model, "pretrained_sst2")
        self.assertEqual(
            QARequest(question="What?", context="A passage.").attribution_model,
            "pretrained_sst2",
        )

    def test_text_requests_reject_removed_tfidf_option(self):
        with self.assertRaises(ValueError):
            SummarizeRequest(text="A passage.", attribution_model="tfidf")

    def test_tfidf_remains_the_default_alignment_scorer(self):
        score = _alignment_scorer("Mitochondria produce ATP.", "ATP production")

        self.assertIsNotNone(score)
        self.assertGreater(score(["Mitochondria produce ATP."])[0], 0.0)

    @patch("attribution.score_evidence_pairs", return_value=np.array([0.8, 0.2]))
    def test_classifier_scorer_uses_source_and_reference_pairs(self, score_pairs):
        score = _alignment_scorer(
            "Evidence sentence. Another sentence.",
            "the summary",
            model_type="finetuned_sst2",
        )

        values = score(["Evidence sentence.", "Another sentence."])

        self.assertEqual(values.tolist(), [0.8, 0.2])
        score_pairs.assert_called_once_with(
            ["Evidence sentence.", "Another sentence."],
            "the summary",
            model_type="finetuned_sst2",
        )

    @patch("attribution.score_evidence_pairs", return_value=np.array([0.75, 0.25]))
    def test_sentence_ranking_uses_selected_classifier(self, score_pairs):
        result = get_sentence_attribution(
            "Plants use light. Ocean currents move heat.",
            "plants use light",
            model_type="finetuned_sst2",
        )

        self.assertEqual(result[0]["sentence"], "Plants use light.")
        self.assertEqual(result[0]["score"], 0.75)
        score_pairs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
