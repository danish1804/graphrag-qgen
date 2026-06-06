"""Unit tests for document deduplication (hash + similarity)."""

import os
import tempfile
import unittest

from utils.content_hash import sha256_bytes, sha256_text
from utils.text_normalization import normalize_document_text
from utils.text_similarity import similarity_ratio


class TestContentHash(unittest.TestCase):
    def test_sha256_bytes_stable(self):
        data = b"hello graphrag"
        self.assertEqual(
            sha256_bytes(data),
            sha256_bytes(data),
        )

    def test_sha256_file_matches_bytes(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"sample pdf content for hash test")
            path = tmp.name
        try:
            from utils.content_hash import sha256_file

            with open(path, "rb") as handle:
                raw = handle.read()
            self.assertEqual(sha256_file(path), sha256_bytes(raw))
        finally:
            os.unlink(path)


class TestTextSimilarity(unittest.TestCase):
    def test_identical_normalized_text_scores_one(self):
        text = "Transaction isolation and deadlock prevention in DBMS."
        score = similarity_ratio(text, text, sample_chars=5000)
        self.assertGreaterEqual(score, 0.99)

    def test_minor_edit_still_high_similarity(self):
        a = "Chapter 1: Database transactions and locking protocols for safety."
        b = "Chapter 2: Database transactions and locking protocols for safety."
        score = similarity_ratio(a, b, sample_chars=5000)
        self.assertGreater(score, 0.85)

    def test_different_topics_low_similarity(self):
        a = "Convolutional neural networks and pooling layers."
        b = "SQL join algorithms and B-tree indexing structures."
        score = similarity_ratio(a, b, sample_chars=5000)
        self.assertLess(score, 0.5)

    def test_text_hash_matches_normalized(self):
        text = "Same content for fingerprinting."
        h1 = sha256_text(normalize_document_text(text))
        h2 = sha256_text(normalize_document_text(text + "  "))
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
