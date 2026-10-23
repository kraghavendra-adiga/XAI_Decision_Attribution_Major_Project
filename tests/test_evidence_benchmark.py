import io
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evidence_benchmark import (
    binary_metrics,
    load_scifact_archive,
    make_evidence_pairs,
    tfidf_similarity_scores,
)


class EvidenceBenchmarkTests(unittest.TestCase):
    def test_builds_relevant_pairs_from_support_and_contradiction_rationales(self):
        claims = [{
            "claim": "A claim about plants.",
            "cited_doc_ids": [1],
            "evidence": {
                "1": [
                    {"label": "SUPPORT", "sentences": [0]},
                    {"label": "CONTRADICT", "sentences": [2]},
                ]
            },
        }]
        corpus = {
            "1": {
                "abstract": [
                    "The first sentence supports the claim.",
                    "This sentence is unrelated.",
                    "The third sentence contradicts the claim.",
                ]
            }
        }

        pairs = make_evidence_pairs(claims, corpus)

        self.assertCountEqual([pair.label for pair in pairs], [1, 1, 0])
        self.assertEqual(len(tfidf_similarity_scores(pairs)), len(pairs))

    def test_loads_expected_files_from_dataset_archive(self):
        contents = {
            "data/corpus.jsonl": json.dumps({"doc_id": 1, "abstract": ["Evidence."]}),
            "data/claims_train.jsonl": json.dumps({"claim": "train", "evidence": {}}),
            "data/claims_dev.jsonl": json.dumps({"claim": "dev", "evidence": {}}),
        }
        with TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "scifact.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, contents_text in contents.items():
                    data = contents_text.encode("utf-8")
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))

            corpus, claims = load_scifact_archive(archive_path)

        self.assertIn("1", corpus)
        self.assertEqual(claims["train"][0]["claim"], "train")
        self.assertEqual(claims["dev"][0]["claim"], "dev")

    def test_binary_metrics_use_continuous_scores_for_auc(self):
        result = binary_metrics([0, 1, 0, 1], [0.1, 0.9, 0.6, 0.8])

        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
