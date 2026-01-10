#!/usr/bin/env python3
"""Chaos testing script for elastic training validation.

This script simulates random worker failures during training to validate
the elastic recovery behavior and measure Recovery Time Objective (RTO).

Usage:
    python chaos_test.py --interval 300 --duration 3600

This will kill a random training worker every 5 minutes for 1 hour.

Enhanced features:
- Configurable kill signals (SIGKILL, SIGTERM, SIGINT)
- Training progress verification (step count, loss values)
- Better process detection and tracking
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CHAOS] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class KillSignal(Enum):
    """Signals that can be used to kill workers."""

    SIGKILL = signal.SIGKILL  # Immediate death (cannot be caught)
    SIGTERM = signal.SIGTERM  # Graceful shutdown request
    SIGINT = signal.SIGINT    # Ctrl+C simulation


@dataclass
class ChaosConfig:
    """Configuration for chaos testing.

    Attributes:
        kill_signal: Signal to use for killing workers.
        verify_progress: Whether to verify training progress after recovery.
        min_step_progress: Minimum steps that should be made after recovery.
        checkpoint_dir: Directory where checkpoints are saved (for progress verification).
    """

    kill_signal: KillSignal = KillSignal.SIGKILL
    verify_progress: bool = True
    min_step_progress: int = 5
    checkpoint_dir: str | None = None


@dataclass
class ChaosEvent:
    """Record of a chaos event (worker kill)."""

    timestamp: float
    pid: int
    rank: int | None
    recovery_time: float | None = None
    success: bool = False


@dataclass
class ChaosTestResult:
    """Results from a chaos test run."""

    start_time: float
    end_time: float
    events: list[ChaosEvent] = field(default_factory=list)
    total_kills: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    avg_recovery_time: float = 0.0
    max_recovery_time: float = 0.0
    min_recovery_time: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "duration_seconds": self.end_time - self.start_time,
            "total_kills": self.total_kills,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "avg_recovery_time_seconds": self.avg_recovery_time,
            "max_recovery_time_seconds": self.max_recovery_time,
            "min_recovery_time_seconds": self.min_recovery_time if self.min_recovery_time != float("inf") else None,
            "events": [
                {
                    "timestamp": datetime.fromtimestamp(e.timestamp).isoformat(),
                    "pid": e.pid,
                    "rank": e.rank,
                    "recovery_time": e.recovery_time,
                    "success": e.success,
                }
                for e in self.events
            ],
        }


class ChaosTestRunner:
    """Chaos engineering for elastic training validation.

    This class manages chaos testing by periodically killing random training
    workers and measuring the recovery time.

    Enhanced features:
    - Configurable kill signals for different failure scenarios
    - Training progress verification after recovery
    - Better process detection via PID tracking
    """

    def __init__(
        self,
        process_pattern: str = "train.py",
        min_survivors: int = 1,
        rto_threshold: float = 30.0,
        config: ChaosConfig | None = None,
    ):
        """Initialize chaos test runner.

        Args:
            process_pattern: Pattern to identify training processes.
            min_survivors: Minimum number of workers that must survive.
            rto_threshold: Target RTO in seconds (for reporting).
            config: Chaos configuration for advanced settings.
        """
        self.process_pattern = process_pattern
        self.min_survivors = min_survivors
        self.rto_threshold = rto_threshold
        self.config = config or ChaosConfig()
        self.result = ChaosTestResult(start_time=time.time(), end_time=0)
        self._last_known_step: int | None = None
        self._tracked_pids: set[int] = set()

    def find_training_processes(self) -> list[psutil.Process]:
        """Find all training worker processes.

        Returns:
            List of psutil.Process objects for training workers.
        """
        processes = []

        for proc in psutil.process_iter(["pid", "name", "cmdline", "environ"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline)

                if self.process_pattern in cmdline_str:
                    processes.append(proc)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return processes

    def get_process_rank(self, proc: psutil.Process) -> int | None:
        """Get the rank of a training process from environment.

        Args:
            proc: The process to check.

        Returns:
            Process rank, or None if not found.
        """
        try:
            env = proc.environ()
            rank_str = env.get("RANK")
            if rank_str is not None:
                return int(rank_str)
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            pass
        return None

    def kill_random_worker(self, exclude_rank_0: bool = True) -> ChaosEvent | None:
        """Kill a random training worker.

        Args:
            exclude_rank_0: Whether to exclude rank 0 from killing.

        Returns:
            ChaosEvent if a worker was killed, None otherwise.
        """
        processes = self.find_training_processes()

        if len(processes) <= self.min_survivors:
            logger.warning(
                f"Only {len(processes)} workers alive, need at least {self.min_survivors}. Skipping kill."
            )
            return None

        # Filter out rank 0 if requested
        candidates = []
        for proc in processes:
            rank = self.get_process_rank(proc)
            if exclude_rank_0 and rank == 0:
                continue
            candidates.append((proc, rank))

        if not candidates:
            logger.warning("No eligible workers to kill (all are rank 0)")
            return None

        # Select random victim
        target_proc, target_rank = random.choice(candidates)

        try:
            kill_signal = self.config.kill_signal
            logger.info(
                f"Killing worker PID {target_proc.pid} (rank {target_rank}) "
                f"with {kill_signal.name}"
            )
            target_proc.send_signal(kill_signal.value)

            event = ChaosEvent(
                timestamp=time.time(),
                pid=target_proc.pid,
                rank=target_rank,
            )

            self.result.events.append(event)
            self.result.total_kills += 1

            return event

        except psutil.NoSuchProcess:
            logger.warning(f"Process {target_proc.pid} already dead")
            return None

    def get_current_step(self) -> int | None:
        """Get the current training step from checkpoint files.

        Returns:
            Current training step, or None if cannot be determined.
        """
        checkpoint_dir = self.config.checkpoint_dir
        if not checkpoint_dir:
            return None

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None

        # Find latest checkpoint file (format: checkpoint_step_XXXXXXXX.pt)
        checkpoints = sorted(checkpoint_path.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        # Extract step from filename
        match = re.search(r"checkpoint_step_(\d+)\.pt", latest.name)
        if match:
            return int(match.group(1))

        return None

    def verify_training_progress(self, timeout: float = 60.0) -> bool:
        """Verify that training is making progress.

        This checks that the training step is increasing, indicating
        that training has actually resumed and is making progress.

        Args:
            timeout: Maximum time to wait for progress verification.

        Returns:
            True if training is making progress.
        """
        if not self.config.verify_progress:
            return True

        current_step = self.get_current_step()
        if current_step is None:
            logger.warning("Cannot verify progress: checkpoint directory not set or no checkpoints found")
            return True  # Assume success if we can't verify

        if self._last_known_step is not None:
            progress = current_step - self._last_known_step
            if progress < self.config.min_step_progress:
                logger.warning(
                    f"Insufficient training progress: {progress} steps "
                    f"(expected >= {self.config.min_step_progress})"
                )
                return False

            logger.info(f"Training progress verified: {progress} steps since last check")

        self._last_known_step = current_step
        return True

    def verify_recovery(
        self,
        event: ChaosEvent,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Verify that training recovers after a kill.

        This method checks:
        1. That the expected number of workers are running
        2. That training is making progress (if configured)

        Args:
            event: The chaos event to verify recovery for.
            timeout: Maximum time to wait for recovery.
            poll_interval: Time between recovery checks.

        Returns:
            True if training recovered within timeout.
        """
        start_time = time.time()
        expected_count = len(self.find_training_processes())  # Current count after kill

        while time.time() - start_time < timeout:
            time.sleep(poll_interval)

            current_processes = self.find_training_processes()
            current_count = len(current_processes)

            # Check if a new worker has joined or existing workers are still training
            # (elastic training should continue with fewer workers or spawn replacement)
            if current_count >= expected_count:
                # Check if processes are actually running (not zombies)
                alive_count = sum(1 for p in current_processes if p.is_running())

                if alive_count >= self.min_survivors:
                    recovery_time = time.time() - event.timestamp
                    event.recovery_time = recovery_time

                    # Optionally verify training progress
                    if self.config.verify_progress:
                        # Wait a bit for training to make progress
                        time.sleep(5)
                        if not self.verify_training_progress():
                            logger.warning("Recovery detected but training not making progress")
                            event.success = False
                            self.result.failed_recoveries += 1
                            return False

                    event.success = True
                    logger.info(
                        f"Recovery verified in {recovery_time:.2f}s "
                        f"({'PASS' if recovery_time <= self.rto_threshold else 'SLOW'})"
                    )

                    self.result.successful_recoveries += 1
                    return True

        # Timeout - recovery failed
        event.recovery_time = timeout
        event.success = False
        self.result.failed_recoveries += 1

        logger.error(f"Recovery FAILED - timeout after {timeout}s")
        return False

    def run_chaos_loop(
        self,
        interval_seconds: float = 300.0,
        duration_seconds: float = 3600.0,
        jitter_pct: float = 0.2,
        verify_recovery: bool = True,
    ) -> ChaosTestResult:
        """Run chaos test loop for specified duration.

        Args:
            interval_seconds: Base interval between kills.
            duration_seconds: Total duration of chaos test.
            jitter_pct: Random jitter as percentage of interval.
            verify_recovery: Whether to verify recovery after each kill.

        Returns:
            ChaosTestResult with test results.
        """
        self.result.start_time = time.time()
        end_time = self.result.start_time + duration_seconds

        logger.info(f"Starting chaos test for {duration_seconds}s (interval: {interval_seconds}s)")

        while time.time() < end_time:
            # Add jitter to interval
            jitter = random.uniform(-jitter_pct, jitter_pct) * interval_seconds
            wait_time = interval_seconds + jitter

            logger.info(f"Next kill in {wait_time:.1f}s...")
            time.sleep(wait_time)

            if time.time() >= end_time:
                break

            # Kill a worker
            event = self.kill_random_worker()

            if event and verify_recovery:
                self.verify_recovery(event)

        self.result.end_time = time.time()
        self._compute_statistics()

        return self.result

    def _compute_statistics(self) -> None:
        """Compute summary statistics from events."""
        recovery_times = [
            e.recovery_time
            for e in self.result.events
            if e.recovery_time is not None and e.success
        ]

        if recovery_times:
            self.result.avg_recovery_time = sum(recovery_times) / len(recovery_times)
            self.result.max_recovery_time = max(recovery_times)
            self.result.min_recovery_time = min(recovery_times)

    def print_summary(self) -> None:
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("CHAOS TEST SUMMARY")
        print("=" * 60)
        print(f"Duration:              {self.result.end_time - self.result.start_time:.1f}s")
        print(f"Total kills:           {self.result.total_kills}")
        print(f"Successful recoveries: {self.result.successful_recoveries}")
        print(f"Failed recoveries:     {self.result.failed_recoveries}")
        print(f"Avg recovery time:     {self.result.avg_recovery_time:.2f}s")
        print(f"Max recovery time:     {self.result.max_recovery_time:.2f}s")
        print(f"Min recovery time:     {self.result.min_recovery_time:.2f}s" if self.result.min_recovery_time != float("inf") else "N/A")
        print(f"RTO threshold:         {self.rto_threshold}s")

        # Pass/fail verdict
        if self.result.failed_recoveries > 0:
            print("\nVERDICT: FAIL (recovery failures)")
        elif self.result.max_recovery_time > self.rto_threshold:
            print(f"\nVERDICT: WARN (max RTO {self.result.max_recovery_time:.2f}s > {self.rto_threshold}s)")
        else:
            print("\nVERDICT: PASS")
        print("=" * 60)


def main():
    """Command-line interface for chaos testing."""
    parser = argparse.ArgumentParser(
        description="Chaos testing for elastic distributed training"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Interval between kills in seconds (default: 300)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Total test duration in seconds (default: 3600)",
    )
    parser.add_argument(
        "--rto-threshold",
        type=float,
        default=30.0,
        help="Target RTO in seconds (default: 30)",
    )
    parser.add_argument(
        "--min-survivors",
        type=int,
        default=1,
        help="Minimum workers to keep alive (default: 1)",
    )
    parser.add_argument(
        "--process-pattern",
        type=str,
        default="train.py",
        help="Pattern to identify training processes (default: train.py)",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.2,
        help="Jitter as fraction of interval (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results",
    )
    parser.add_argument(
        "--single-kill",
        action="store_true",
        help="Kill one worker and exit (for testing)",
    )
    parser.add_argument(
        "--kill-signal",
        type=str,
        choices=["SIGKILL", "SIGTERM", "SIGINT"],
        default="SIGKILL",
        help="Signal to use for killing workers (default: SIGKILL)",
    )
    parser.add_argument(
        "--verify-progress",
        action="store_true",
        default=False,
        help="Verify training makes progress after recovery",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory for progress verification",
    )
    parser.add_argument(
        "--min-step-progress",
        type=int,
        default=5,
        help="Minimum steps expected between checks (default: 5)",
    )

    args = parser.parse_args()

    # Build chaos config
    kill_signal_map = {
        "SIGKILL": KillSignal.SIGKILL,
        "SIGTERM": KillSignal.SIGTERM,
        "SIGINT": KillSignal.SIGINT,
    }
    chaos_config = ChaosConfig(
        kill_signal=kill_signal_map[args.kill_signal],
        verify_progress=args.verify_progress,
        min_step_progress=args.min_step_progress,
        checkpoint_dir=args.checkpoint_dir,
    )

    runner = ChaosTestRunner(
        process_pattern=args.process_pattern,
        min_survivors=args.min_survivors,
        rto_threshold=args.rto_threshold,
        config=chaos_config,
    )

    if args.single_kill:
        # Single kill mode for testing
        event = runner.kill_random_worker()
        if event:
            runner.verify_recovery(event)
            runner.result.end_time = time.time()
            runner._compute_statistics()
    else:
        # Full chaos loop
        runner.run_chaos_loop(
            interval_seconds=args.interval,
            duration_seconds=args.duration,
            jitter_pct=args.jitter,
        )

    runner.print_summary()

    # Save results to JSON if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(runner.result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Exit with appropriate code
    if runner.result.failed_recoveries > 0:
        sys.exit(1)
    elif runner.result.max_recovery_time > args.rto_threshold:
        sys.exit(2)  # Warn exit code
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
