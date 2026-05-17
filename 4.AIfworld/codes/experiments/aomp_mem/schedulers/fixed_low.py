"""Fixed-low schedule."""

from __future__ import annotations

from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext, SchedulerDecision


class FixedLowScheduler(BaseScheduler):
    def __init__(self, *, interval: int = 5, n_cal_high: int = 3) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        if n_cal_high < 0:
            raise ValueError(f"n_cal_high must be non-negative, got {n_cal_high}")
        self.interval = int(interval)
        self.n_cal_high = int(n_cal_high)

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        should_refine = context.task_index % self.interval == 0
        return SchedulerDecision(
            n_refine=self.n_cal_high if should_refine else 0,
            drift_detected=False,
            metadata={"schedule": "fixed_low", "interval": self.interval},
        )
