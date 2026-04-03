from __future__ import annotations
import numpy as np

class MarkerTracker:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.pos: np.ndarray | None = None
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.last_valid_pos: np.ndarray | None = None
        self.missed_frames = 0

    def predict(self) -> np.ndarray | None:
        if self.pos is None:
            return None
        return self.pos + self.vel

    def update(self, measurement: np.ndarray | None, confidence: float = 1.0):
        if measurement is None:
            if self.pos is not None:
                self.pos = self.pos + self.vel
                self.vel *= 0.95
            self.missed_frames += 1
            return

        if self.pos is None:
            self.pos = measurement
            self.vel = np.array([0.0, 0.0], dtype=np.float32)
            self.missed_frames = 0
            return

        pred = self.pos + self.vel
        res = measurement - pred

        a = self.alpha * confidence
        b = self.beta * confidence

        self.pos = pred + a * res
        self.vel = self.vel + b * res
        self.missed_frames = 0
        self.last_valid_pos = self.pos.copy()
