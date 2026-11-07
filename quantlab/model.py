"""A tiny linear classifier trained in float, then quantized.

The model is a single linear layer over hashed bag-of-words features trained
with perceptron updates on a deterministic synthetic dataset - small enough
that the accuracy cost of int8 weight quantization is directly measurable,
and seeded so a config reproduces the run bit for bit.
"""

from __future__ import annotations

import random

from .quantize import dequantize, quantize_per_channel, quantize_tensor

TOPICS = {
    "space": ["orbit", "rocket", "lunar", "galaxy", "satellite", "cosmos"],
    "food": ["recipe", "flavor", "bake", "spice", "kitchen", "taste"],
    "sport": ["match", "score", "coach", "league", "tournament", "athlete"],
}


def build_dataset(seed: int = 0, per_topic: int = 40, vocab_size: int = 500) -> list[dict]:
    """Each sample draws mostly from its topic's signature words plus noise."""
    rng = random.Random(seed)
    samples = []
    topics = sorted(TOPICS)
    for topic in topics:
        sig = TOPICS[topic]
        for _ in range(per_topic):
            words = [rng.choice(sig) for _ in range(6)]
            words += [f"noise{rng.randrange(60)}" for _ in range(4)]
            rng.shuffle(words)
            samples.append({"words": words, "label": topics.index(topic)})
    rng.shuffle(samples)
    return samples, topics


def _bucket(word: str, dim: int) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "big") % dim


def featurize(words: list[str], dim: int) -> list[float]:
    vec = [0.0] * dim
    for w in words:
        vec[_bucket(w, dim)] += 1.0
    return vec


class LinearClassifier:
    def __init__(self, dim: int, classes: int) -> None:
        self.dim = dim
        self.classes = classes
        self.weights = [[0.0] * dim for _ in range(classes)]

    def scores(self, x: list[float]) -> list[float]:
        return [sum(w * xi for w, xi in zip(row, x)) for row in self.weights]

    def predict(self, x: list[float]) -> int:
        s = self.scores(x)
        return max(range(self.classes), key=lambda c: s[c])

    def train(self, samples: list[dict], epochs: int = 5) -> "LinearClassifier":
        for _ in range(epochs):
            for s in samples:
                x = featurize(s["words"], self.dim)
                pred = self.predict(x)
                if pred != s["label"]:
                    for j, xi in enumerate(x):
                        self.weights[s["label"]][j] += xi
                        self.weights[pred][j] -= xi
        return self

    def accuracy(self, samples: list[dict]) -> float:
        correct = sum(1 for s in samples if self.predict(featurize(s["words"], self.dim)) == s["label"])
        return correct / max(len(samples), 1)

    def quantized(self, mode: str = "per-channel") -> "QuantizedClassifier":
        if mode == "per-tensor":
            packed = quantize_tensor(self.weights)
        elif mode == "per-channel":
            packed = quantize_per_channel(self.weights)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return QuantizedClassifier(packed, self.dim, self.classes)


class QuantizedClassifier:
    """Stores int8 weights and dequantizes for inference - the shape a real
    quantized deployment takes."""

    def __init__(self, packed: dict, dim: int, classes: int) -> None:
        self.packed = packed
        self.dim = dim
        self.classes = classes
        self._weights = dequantize(packed)

    def predict(self, x: list[float]) -> int:
        s = [sum(w * xi for w, xi in zip(row, x)) for row in self._weights]
        return max(range(self.classes), key=lambda c: s[c])

    def accuracy(self, samples: list[dict]) -> float:
        correct = sum(1 for s in samples if self.predict(featurize(s["words"], self.dim)) == s["label"])
        return correct / max(len(samples), 1)
