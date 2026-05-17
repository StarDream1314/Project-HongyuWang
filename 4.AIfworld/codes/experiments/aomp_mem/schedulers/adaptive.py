"""Adaptive burst schedule."""

from __future__ import annotations

from typing import Optional

import numpy as np

from experiments.aomp_mem.certificate.cusum_fdr import CUSUMFDRGate
from experiments.aomp_mem.certificate.feature_pipeline import BurstHistoryState, compute_features
from experiments.aomp_mem.certificate.mqc import MQCCertificate
from experiments.aomp_mem.certificate.ole_certificate import OLECertificate
from experiments.aomp_mem.core.drift import detect_performance_drift, detect_similarity_warning
from experiments.aomp_mem.evaluation.memory_quality import compute_memory_quality
from experiments.aomp_mem.schedulers.base import BaseScheduler, SchedulerContext, SchedulerDecision


class AdaptiveScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        warmup: int = 20,
        drift_window: int = 5,
        drift_threshold: float = 0.2,
        burst_length: int = 10,
        n_cal_low: int = 0,
        n_cal_high: int = 3,
        similarity_threshold: float = 0.3,
        anchors: Optional[np.ndarray] = None,
        mqc_certificate: Optional[MQCCertificate] = None,
        ole_certificate: Optional[OLECertificate] = None,
        cusum_gate: Optional[CUSUMFDRGate] = None,
        lambda_fresh: float = 0.20,
        coverage_radius: Optional[float] = None,
    ) -> None:
        self.warmup = int(warmup)
        self.drift_window = int(drift_window)
        self.drift_threshold = float(drift_threshold)
        self.burst_length = int(burst_length)
        self.n_cal_low = int(n_cal_low)
        self.n_cal_high = int(n_cal_high)
        self.similarity_threshold = float(similarity_threshold)
        self.anchors = None if anchors is None else np.asarray(anchors, dtype=float)
        self.mqc_cert = mqc_certificate
        self.ole_cert = ole_certificate
        self.cusum_gate = cusum_gate or (CUSUMFDRGate() if mqc_certificate is not None else None)
        self.lambda_fresh = float(lambda_fresh)
        self.coverage_radius = None if coverage_radius is None else float(coverage_radius)
        self._burst_remaining = 0
        self._last_burst_step: Optional[int] = None
        self._burst_steps: list[int] = []

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        drift_detected, baseline, current = detect_performance_drift(
            success_history=context.success_history,
            warmup=self.warmup,
            window=self.drift_window,
            drift_threshold=self.drift_threshold,
        )

        similarity_warning = False
        if context.query_embedding is not None and context.history_embeddings:
            similarity_warning = detect_similarity_warning(
                query_embedding=context.query_embedding,
                history_embeddings=context.history_embeddings,
                similarity_threshold=self.similarity_threshold,
            )

        if self._can_use_ole(context):
            return self._decide_with_ole(
                context=context,
                baseline=baseline,
                current=current,
                similarity_warning=similarity_warning,
            )

        if self._can_use_mqc(context):
            return self._decide_with_mqc(
                context=context,
                baseline=baseline,
                current=current,
                similarity_warning=similarity_warning,
            )

        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            return SchedulerDecision(
                n_refine=self.n_cal_high,
                drift_detected=drift_detected,
                metadata={
                    "schedule": "adaptive",
                    "baseline_success_rate": baseline,
                    "current_success_rate": current,
                    "burst_remaining": self._burst_remaining,
                    "similarity_warning": similarity_warning,
                },
            )

        if drift_detected:
            self._burst_remaining = max(0, self.burst_length - 1)
            return SchedulerDecision(
                n_refine=self.n_cal_high,
                drift_detected=True,
                metadata={
                    "schedule": "adaptive",
                    "baseline_success_rate": baseline,
                    "current_success_rate": current,
                    "burst_remaining": self._burst_remaining,
                    "similarity_warning": similarity_warning,
                },
            )

        return SchedulerDecision(
            n_refine=self.n_cal_low,
            drift_detected=False,
            metadata={
                "schedule": "adaptive",
                "baseline_success_rate": baseline,
                "current_success_rate": current,
                "burst_remaining": self._burst_remaining,
                "similarity_warning": similarity_warning,
            },
        )

    def _can_use_ole(self, context: SchedulerContext) -> bool:
        return (
            self.ole_cert is not None
            and self.anchors is not None
            and context.query_embedding is not None
            and context.memory_entries is not None
            and context.retrieved is not None
        )

    def _can_use_mqc(self, context: SchedulerContext) -> bool:
        return (
            self.mqc_cert is not None
            and self.anchors is not None
            and context.memory_entries is not None
            and context.retrieved is not None
        )

    def _decide_with_ole(
        self,
        *,
        context: SchedulerContext,
        baseline: float,
        current: Optional[float],
        similarity_warning: bool,
    ) -> SchedulerDecision:
        assert self.ole_cert is not None
        assert self.anchors is not None

        entries = context.memory_entries or []
        retrieved = context.retrieved or []
        metrics = compute_memory_quality(
            entries=entries,
            retrieved=retrieved,
            anchors=self.anchors,
            task_step=context.task_index,
            lambda_fresh=self.lambda_fresh,
            coverage_radius=self.coverage_radius,
        )
        burst_state = BurstHistoryState(
            last_burst_step=self._last_burst_step,
            burst_count_in_window=self._burst_count_in_window(context.task_index),
        )
        raw_features = compute_features(
            metrics,
            query_embedding=context.query_embedding,
            memory_entries=entries,
            retrieved=retrieved,
            anchors=self.anchors,
            burst_state=burst_state,
            task_step=context.task_index,
        )
        result = self.ole_cert.compute(raw_features)

        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            n_refine = self.n_cal_high
        elif result.triggered:
            self._burst_remaining = max(0, self.burst_length - 1)
            n_refine = self.n_cal_high
            self._record_burst(context.task_index)
        else:
            n_refine = self.n_cal_low

        metadata = {
            "schedule": "adaptive",
            "baseline_success_rate": baseline,
            "current_success_rate": current,
            "burst_remaining": self._burst_remaining,
            "similarity_warning": similarity_warning,
            "mqc_t": result.mean,
            "mqc_mode": "ole",
            "ole_mean_t": result.mean,
            "ole_var_t": result.var,
            "ole_ucb_t": result.ucb,
            "ole_kappa_t": self.ole_cert.kappa,
            "coverage_t": metrics.coverage,
            "precision_t": metrics.precision,
            "redundancy_t": metrics.redundancy,
            "freshness_t": metrics.freshness,
            "no_retrieval_window_flag_t": metrics.no_retrieval_window_flag,
            "drift_onset_flag_t": result.triggered,
        }
        return SchedulerDecision(
            n_refine=n_refine,
            drift_detected=result.triggered,
            metadata=metadata,
        )

    def _decide_with_mqc(
        self,
        *,
        context: SchedulerContext,
        baseline: float,
        current: Optional[float],
        similarity_warning: bool,
    ) -> SchedulerDecision:
        assert self.mqc_cert is not None
        assert self.anchors is not None

        metrics = compute_memory_quality(
            entries=context.memory_entries or [],
            retrieved=context.retrieved or [],
            anchors=self.anchors,
            task_step=context.task_index,
            lambda_fresh=self.lambda_fresh,
            coverage_radius=self.coverage_radius,
        )
        result = self.mqc_cert.compute(metrics, budget_remaining=self._remaining_budget(context))
        drift_onset = False
        cusum_stat = None
        if self.cusum_gate is not None:
            drift_onset = self.cusum_gate.update(result.gap_upper - self.mqc_cert.tau, sigma=1.0)
            cusum_stat = self.cusum_gate.cusum_stat
            if drift_onset:
                self.mqc_cert.recalibrate_flag = True

        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            n_refine = self.n_cal_high
        elif result.triggered:
            self._burst_remaining = max(0, self.burst_length - 1)
            n_refine = self.n_cal_high
        else:
            n_refine = self.n_cal_low

        metadata = {
            "schedule": "adaptive",
            "baseline_success_rate": baseline,
            "current_success_rate": current,
            "burst_remaining": self._burst_remaining,
            "similarity_warning": similarity_warning,
            "mqc_t": result.gap_upper,
            "mqc_mode": result.mode,
            "coverage_t": metrics.coverage,
            "precision_t": metrics.precision,
            "redundancy_t": metrics.redundancy,
            "freshness_t": metrics.freshness,
            "no_retrieval_window_flag_t": metrics.no_retrieval_window_flag,
            "drift_onset_flag_t": drift_onset,
        }
        if cusum_stat is not None:
            metadata["cusum_stat_t"] = cusum_stat
        return SchedulerDecision(
            n_refine=n_refine,
            drift_detected=result.triggered,
            metadata=metadata,
        )

    @staticmethod
    def _remaining_budget(_context: SchedulerContext) -> float:
        return 1.0

    def _burst_count_in_window(self, task_index: int) -> int:
        window_start = int(task_index) - int(self.burst_length)
        return sum(1 for step in self._burst_steps if window_start <= step < int(task_index))

    def _record_burst(self, task_index: int) -> None:
        self._last_burst_step = int(task_index)
        self._burst_steps.append(int(task_index))
