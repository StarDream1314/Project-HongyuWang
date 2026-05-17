"""Oracle-high schedule."""

from __future__ import annotations

from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext, SchedulerDecision


class OracleHighScheduler(BaseScheduler):
    def __init__(self, *, n_cal_high: int = 3) -> None:
        if n_cal_high < 0:
            raise ValueError(f"n_cal_high must be non-negative, got {n_cal_high}")
        self.n_cal_high = int(n_cal_high)

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        return SchedulerDecision(
            n_refine=self.n_cal_high,
            drift_detected=False,
            metadata={"schedule": "oracle_high"},
        )
