import unittest

from quantlab.analysis import compare_modes, imbalanced_matrix
from quantlab.model import LinearClassifier, build_dataset, featurize


class DatasetTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_dataset(seed=1), build_dataset(seed=1))

    def test_seed_changes_data(self):
        self.assertNotEqual(build_dataset(seed=1), build_dataset(seed=2))

    def test_labels_cover_topics(self):
        samples, topics = build_dataset(seed=0, per_topic=5)
        self.assertEqual({s["label"] for s in samples}, {0, 1, 2})


class FeaturizeTests(unittest.TestCase):
    def test_deterministic_across_calls(self):
        self.assertEqual(featurize(["orbit", "recipe"], 32), featurize(["orbit", "recipe"], 32))

    def test_bag_of_words_counts(self):
        v = featurize(["orbit", "orbit", "recipe"], 32)
        self.assertEqual(sum(v), 3.0)
        self.assertLessEqual(max(v), 2.0)


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.train, _ = build_dataset(seed=0, per_topic=40)
        self.test, _ = build_dataset(seed=1, per_topic=20)
        self.clf = LinearClassifier(dim=64, classes=3).train(self.train, epochs=5)

    def test_trains_to_high_accuracy(self):
        self.assertGreaterEqual(self.clf.accuracy(self.test), 0.9)

    def test_int8_matches_fp32_accuracy_on_fixture(self):
        for mode in ("per-tensor", "per-channel"):
            q = self.clf.quantized(mode)
            self.assertEqual(q.accuracy(self.test), self.clf.accuracy(self.test))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.clf.quantized("int4")

    def test_deterministic_training(self):
        other = LinearClassifier(dim=64, classes=3).train(self.train, epochs=5)
        self.assertEqual(other.weights, self.clf.weights)


class AnalysisTests(unittest.TestCase):
    def test_per_channel_wins_small_rows(self):
        rep = compare_modes(imbalanced_matrix())
        self.assertGreater(rep["small_row_snr_gain_db"], 10.0)
        self.assertGreater(rep["per_channel"]["small_row_snr_db"], 30.0)

    def test_balanced_matrix_small_gap(self):
        rep = compare_modes(imbalanced_matrix(boost=1.0))
        self.assertLess(rep["small_row_snr_gain_db"], 10.0)


if __name__ == "__main__":
    unittest.main()
