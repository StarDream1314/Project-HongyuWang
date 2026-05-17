"""Exp3-based refinement scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext, SchedulerDecision


@dataclass
class Exp3Observation:
    arm_index: int
    success: bool
    cost: float


class Exp3Scheduler(BaseScheduler):
    def __init__(
        self,
        *,
        n_cal_low: int = 0,
        n_cal_high: int = 3,
        horizon: int = 100,
        lambda_cost: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.arms: List[int] = [0, int(n_cal_low), int(n_cal_high)]
        self.weights = np.ones(len(self.arms), dtype=float)
        self.horizon = max(1, int(horizon))
        self.lambda_cost = float(lambda_cost)
        self.max_cost = max(self.arms) + 2
        self.gamma = min(1.0, math.sqrt(len(self.arms) * math.log(len(self.arms)) / self.horizon))
        self.rng = rng or np.random.default_rng(0)
        self._last_probs = self._compute_probs()
        self._last_choice: Optional[int] = None

    def _compute_probs(self) -> np.ndarray:
        normalized = self.weights / np.sum(self.weights)
        return (1.0 - self.gamma) * normalized + self.gamma / len(self.arms)

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        self._last_probs = self._compute_probs()
        arm_index = int(self.rng.choice(len(self.arms), p=self._last_probs))
        self._last_choice = arm_index
        return SchedulerDecision(
            n_refine=self.arms[arm_index],
            drift_detected=False,
            metadata={
                "schedule": "exp3",
                "arm_index": arm_index,
                "probabilities": self._last_probs.tolist(),
            },
        )

    def observe(self, *, success: bool, cost: float, arm_index: Optional[int] = None) -> Exp3Observation:
        if arm_index is None:
            if self._last_choice is None:
                raise RuntimeError("observe() called before decide()")
            arm_index = self._last_choice
        probability = float(self._last_probs[arm_index])
        loss = (1.0 - float(success)) + self.lambda_cost * (float(cost) / self.max_cost)
        estimated_loss = loss / max(probability, 1e-12)
        self.weights[arm_index] *= math.exp(-self.gamma * estimated_loss / len(self.arms))
        return Exp3Observation(arm_index=arm_index, success=success, cost=cost)
