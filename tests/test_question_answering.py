import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_utils import answer_question
from text_explainer import explain_text_qa


class _FakeEncoding(dict):
    def sequence_ids(self, batch_index):
        return [None, 0, None, 1, 1, None]


class _FakeTokenizer:
    def __init__(self):
        self.options = None

    def __call__(self, question, context, **options):
        self.options = options
        return _FakeEncoding(
            input_ids=torch.tensor(
                [[101, 11, 102, 21, 22, 102], [101, 11, 102, 31, 32, 102]]
            ),
            attention_mask=torch.ones((2, 6), dtype=torch.long),
            overflow_to_sample_mapping=torch.tensor([0, 0]),
        )

    def decode(self, token_ids, skip_special_tokens=True):
        if token_ids.tolist() == [31, 32]:
            return "late-window answer"
        return "early-window distractor"


class _FakeQAModel:
    def __call__(self, input_ids, attention_mask):
        start_logits = torch.full((2, 6), -10.0)
        end_logits = torch.full((2, 6), -10.0)
        start_logits[0, 3] = 4.0
        end_logits[0, 4] = 4.0
        start_logits[1, 3] = 8.0
        end_logits[1, 4] = 8.0
        return SimpleNamespace(start_logits=start_logits, end_logits=end_logits)


class _FakeGemini:
    def __init__(self, reply):
        self.reply = reply
        self.prompt = None

    def generate_content(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(text=json.dumps(self.reply))


class QuestionAnsweringTests(unittest.TestCase):
    def test_answer_question_searches_overlapping_windows_in_long_context(self):
        tokenizer = _FakeTokenizer()
        model = _FakeQAModel()

        with patch("model_utils.get_qa_model", return_value=(model, tokenizer)):
            answer, confidence = answer_question(
                "What is the answer?",
                "An early passage followed by later evidence.",
            )

        self.assertEqual(answer, "late-window answer")
        self.assertGreater(confidence, 0.0)
        self.assertEqual(tokenizer.options["max_length"], 384)
        self.assertEqual(tokenizer.options["stride"], 128)
        self.assertTrue(tokenizer.options["return_overflowing_tokens"])

    def test_hosted_answer_prompt_requests_direct_concise_answers(self):
        source = "Plants use sunlight and carbon dioxide to produce glucose."
        gemini = _FakeGemini({
            "answerable": True,
            "answer": "Photosynthesis produces glucose from sunlight and carbon dioxide.",
            "sources": [{
                "text": source,
                "contribution": 1.0,
                "reason": "It directly states the products.",
            }],
        })

        with (
            patch("text_explainer.gemini_available", return_value=True),
            patch("text_explainer.gemini_model", return_value=gemini),
        ):
            result = explain_text_qa(
                "What does photosynthesis produce?",
                source,
            )

        self.assertEqual(result["engine"], "gemini")
        self.assertIn("Start with a direct answer", gemini.prompt)
        self.assertIn("one to three sentences", gemini.prompt)


if __name__ == "__main__":
    unittest.main()
