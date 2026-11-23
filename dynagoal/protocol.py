"""Clock and scheduling primitives; independent of torch and the simulator."""
from dataclasses import dataclass
import math


TASKS = ("pusht", "cube", "reacher", "tworoom")
TRACKS = ("deadline", "goal_switch")
DEADLINE_FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class Protocol:
    track: str
    feedback_steps: int = 25
    initial_goal_offset: int = 25
    switch_step: int = 25
    switched_goal_offset: int = 50
    deadline_steps: int = 50
    switch_budget_steps: int = 100
    latency_scale: float = 1.0

    def __post_init__(self):
        if self.track not in TRACKS:
            raise ValueError(self.track)
        if not math.isfinite(self.latency_scale) or self.latency_scale < 0:
            raise ValueError("latency_scale must be finite and nonnegative")
        if self.switch_step % self.feedback_steps:
            raise ValueError("Goal notification must coincide with scheduled feedback")


def plan_ready(tick: int, dt: float, ready_at: float) -> bool:
    """Apply a completed plan at the next control boundary, never earlier."""
    return tick * dt + 1e-12 >= ready_at


def deadline_success(success, first_hit_step, dt, cumulative_plan_seconds,
                     deadline_seconds):
    if not success:
        return False
    if first_hit_step is None or first_hit_step < 1 or dt <= 0:
        raise ValueError("Successful episodes need a positive first-hit timestamp")
    return first_hit_step * dt + cumulative_plan_seconds <= deadline_seconds + 1e-12


def shift_history(history, action):
    """Use every action actually executed, including latency hold actions."""
    import numpy as np
    return np.concatenate((history[1:], np.asarray(action)[None]), axis=0)


def goal_margin(task, current, goal):
    """Native success tolerances, used ONLY for outcome-independent case sampling.

    Reacher deliberately retains the original unwrapped qpos comparison;
    changing its topology here would silently change the comparator protocol.
    """
    import numpy as np
    a, b = np.asarray(current), np.asarray(goal)
    if task == "pusht":
        angle = np.abs(a[..., 4] - b[..., 4]) % (2 * np.pi)
        angle = np.minimum(angle, 2 * np.pi - angle)
        return np.maximum(np.linalg.norm(a[..., :4] - b[..., :4], axis=-1) / 20,
                          angle / (np.pi / 9))
    if task == "cube":
        return np.linalg.norm(a - b, axis=-1) / .04
    if task == "reacher":
        return np.max(np.abs(a - b), axis=-1) / .05
    if task == "tworoom":
        return np.linalg.norm(a - b, axis=-1) / 16
    raise ValueError(task)


def initial_hold(task, action_space, relative=True, agent_position=None):
    """Raw ENVIRONMENT action units, not pixels or dataset-standardized units."""
    import numpy as np
    value = np.zeros(action_space.shape, dtype=np.float32)
    if task == "pusht" and not relative:
        value = np.asarray(agent_position, dtype=np.float32).copy()
    if not action_space.contains(value):
        raise ValueError("Initial hold is outside the environment action space")
    return value
