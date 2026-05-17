"""Cheap-only schedule."""

from __future__ import annotations

from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext, SchedulerDecision


class CheapOnlyScheduler(BaseScheduler):
    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        return SchedulerDecision(n_refine=0, drift_detected=False, metadata={"schedule": "cheap_only"})
