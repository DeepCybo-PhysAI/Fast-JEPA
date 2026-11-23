from pathlib import Path
import copy, hashlib, json, time
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
import stable_worldmodel as swm
from stable_worldmodel.world.world import _extract_init_goal, _apply_callables
from stable_worldmodel.policy import BasePolicy
from diagnostics import load
from eval import get_dataset, img_transform
from history_policy import BlockStandardScaler, build_initial_history
from graph_modules import report as graph_report
from graph_solver import GraphSearchSolver
from sdpa_policy import sdpa_kernel_context
from .protocol import Protocol, plan_ready, deadline_success, shift_history, initial_hold, DEADLINE_FACTORS

def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(data), indent=2, allow_nan=False))
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()

def native_dt(task, raw):
    if task == "pusht":
        return 1.0 / raw.control_hz, "native control_hz (not pymunk substep dt)"
    if task == "cube":
        value = raw.control_timestep
        return float(value() if callable(value) else value), "OGBench native control_timestep"
    if task == "reacher":
        return float(raw.dmc_env.control_timestep() * raw.action_repeat), "DMC control_timestep x wrapper action_repeat"
    return 1.0 / raw.metadata["render_fps"], "declared playback/deployment period; TwoRoom has no physical clock"

class Runner:
    def __init__(self, task, arm, cfg, dataset, graph_mode):
        self.task, self.arm, self.cfg, self.dataset = task, arm, cfg, dataset
        self.world = swm.World(**{**OmegaConf.to_container(cfg.world), "num_envs": 1,
                                 "max_episode_steps": 1000}, image_shape=(224, 224))
        self.raw = self.world.envs.envs[0].unwrapped
        self.dt, self.dt_source = native_dt(task, self.raw)
        if arm != 'fast_guarded':
            raise ValueError('This distribution contains Fast-JEPA only')
        self.checkpoint = Path(cfg.checkpoint)
        self.model = load(self.checkpoint).eval().requires_grad_(False)
        if self.model.world_model_type != 'fast':
            raise ValueError('Expected a Fast-JEPA checkpoint')
        self.model.prune_actor_terminal = True
        self.solver = GraphSearchSolver(model=self.model, batch_size=1, device='cuda',
            seed=1, var_scale=.25, num_samples=64, n_steps=2, topk=8,
            prune_actor_terminal=True, cache_encoding=True, cuda_graph_mode='all')
        self.solver.configure(action_space=self.world.envs.action_space, n_envs=1,
                              config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5))
        # Match original all-row normalization exactly; the benchmark does not refit on eval cases.
        self.process = {}
        for col in cfg.dataset.keys_to_cache:
            values = np.asarray(dataset.get_col_data(col))
            values = values[~np.isnan(values).any(axis=1)]
            self.process[col] = StandardScaler().fit(values)
        self.scaler = self.process["action"]
        self.process["action"] = BlockStandardScaler(self.scaler)
        self.preprocessor = BasePolicy()
        self.preprocessor.process = self.process
        self.preprocessor.transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
        self.capture_seconds = 0.0

    def prepared(self, info, history):
        # No simulator physical state, future trajectories, or labels enter the policy.
        return self.preprocessor._prepare_info(dict(
            pixels=np.asarray(info["pixels"]).copy(), goal=np.asarray(info["goal"]).copy(),
            action=history.reshape(1, 1, -1).astype(np.float32)))

    def make_plan(self, info, history):
        torch.cuda.synchronize()
        start = time.perf_counter()
        prepared = self.prepared(info, history)
        output = self.solver.solve(prepared, init_action=None)
        normalized = output["actions"].reshape(25, -1).numpy()
        actions = self.scaler.inverse_transform(normalized).astype(np.float32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not np.isfinite(actions).all():
            raise FloatingPointError("Nonfinite executable plan")
        return actions, elapsed

    def initialize(self, case):
        init, goal, _ = _extract_init_goal(self.dataset, [case["episode"]], [case["start"]], 25)
        _, switched, _ = _extract_init_goal(self.dataset, [case["episode"]], [case["start"]], 50)
        initial_seed = init.get("seed")
        if initial_seed is None:
            initial_seed = [case["env_seed"]]
        self.world.reset(seed=initial_seed)
        callables = OmegaConf.to_container(self.cfg.eval.callables, resolve=True)
        merged = {**init, **goal}
        _apply_callables(self.raw, callables, {k: v[0] for k, v in merged.items()})
        for source in (init, goal):
            for key, value in source.items():
                if key in self.world.infos or key in goal:
                    self.world.infos[key] = value[:, None, ...].copy()
        self.goal = {k: v[:, None, ...].copy() for k, v in goal.items()}
        self.switched = {k: v[:, None, ...].copy() for k, v in switched.items()}
        self.goal_calls = [c for c in callables if c["method"] not in ("_set_state", "set_state")]
        self.history = build_initial_history(self.dataset, [case["episode"]], [case["start"]], 5)[0]
        self.solver.torch_gen.manual_seed(case["plan_seed"])
        self.last_action = initial_hold(self.task, self.world.envs.single_action_space,
                     relative=getattr(self.raw, "relative", True),
                     agent_position=np.asarray(self.raw.agent.position) if self.task == "pusht" else None)
        return self.world.infos

    def change_goal(self):
        # Set only targets; never restore a future physical state.
        before = self.physical_state()
        _apply_callables(self.raw, self.goal_calls, {k: v[0, 0] for k, v in self.switched.items()})
        if not np.array_equal(before, self.physical_state()):
            raise AssertionError("Goal update changed physical agent/object state")
        self.goal = copy.deepcopy(self.switched)
        self.world.infos.update(copy.deepcopy(self.goal))

    def physical_state(self):
        if self.task == "pusht":
            return np.r_[np.asarray(self.raw.agent.position), np.asarray(self.raw.block.position), self.raw.block.angle]
        if self.task == "tworoom":
            return np.asarray(self.raw.agent_position).copy()
        data = self.raw.dmc_env.physics.data if self.task == "reacher" else self.raw._data
        return np.r_[np.asarray(data.qpos), np.asarray(data.qvel)]

    def physical_time(self):
        if self.task == "reacher":
            return float(self.raw.dmc_env.physics.data.time)
        if self.task == "cube":
            return float(self.raw._data.time)
        return None

    def step(self, action):
        before_time = self.physical_time()
        _, _, term, trunc, info = self.world.envs.step(np.asarray(action)[None])
        after_time = self.physical_time()
        if before_time is not None and after_time <= before_time:
            raise AssertionError("Physics clock reset/stalled during a continuous episode")
        self.world.infos = info
        info.update(copy.deepcopy(self.goal))
        self.history = shift_history(self.history, action)
        self.last_action = np.asarray(action).copy()
        return bool(term[0]), bool(trunc[0])

    def warmup(self, case):
        self.initialize(case)
        times = []
        for _ in range(6):
            _, elapsed = self.make_plan(self.world.infos, self.history)
            times.append(elapsed)
        self.capture_seconds = sum(item["capture_seconds"] for item in graph_report(self.model).values())
        return dict(warm_complete_plan_ms=[x * 1000 for x in times[-3:]],
                    graph_capture_seconds=self.capture_seconds,
                    solver_graph_setup_seconds=getattr(self.solver, "graph_setup_seconds", 0.0))

    def gate_native(self, case):
        """Compare the manual zero-delay loop against the unchanged official loop."""
        from history_policy import StatefulActionHistoryPolicy
        manual = self.episode(case, Protocol("deadline"))
        history = build_initial_history(self.dataset, [case["episode"]], [case["start"]], 5)
        policy = swm.policy.WorldModelPolicy(solver=self.solver,
                     config=swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5),
                     process=self.process, transform=self.preprocessor.transform)
        policy = StatefulActionHistoryPolicy(policy, history)
        self.world.set_policy(policy)
        self.solver.torch_gen.manual_seed(case["plan_seed"])
        executed = []
        native_reset, native_step = self.world.reset, self.world.envs.step
        def reset(seed=None, options=None):
            return native_reset(seed=[case["env_seed"]] if seed is None else seed, options=options)
        def step(actions, **kwargs):
            executed.append(np.asarray(actions)[0].copy())
            return native_step(actions, **kwargs)
        self.world.reset, self.world.envs.step = reset, step
        try:
            result = self.world.evaluate(dataset=self.dataset, episodes_idx=[case["episode"]],
                         start_steps=[case["start"]], goal_offset=25, eval_budget=50,
                         callables=OmegaConf.to_container(self.cfg.eval.callables, resolve=True), video=None)
        finally:
            self.world.reset, self.world.envs.step = native_reset, native_step
        expected = np.asarray([row["action"] for row in manual["executed"]], np.float32)
        actual = np.asarray(executed, np.float32)
        if expected.shape != actual.shape or not np.array_equal(expected, actual):
            diff = float(np.max(np.abs(expected-actual))) if expected.shape == actual.shape else None
            raise AssertionError(f"Native action mismatch {self.arm}: manual={expected.shape}, native={actual.shape}, max_abs={diff}")
        if manual["success"] != bool(result["episode_successes"][0]):
            raise AssertionError("Native success mismatch")
        return dict(pass_exact_actions=True, steps=len(actual), success=manual["success"],
                    actions_sha256=hashlib.sha256(actual.tobytes()).hexdigest())

    def episode(self, case, protocol):
        # DMC would otherwise auto-reset after reaching the OLD goal. Suppress
        # only that terminal event before the scheduled switch, not physics.
        self.initialize(case)
        self._suppress_old_goal_end = protocol.track == "goal_switch"
        original = self.raw.dmc_env.task.get_termination if self.task == "reacher" else None
        if original is not None:
            self._native_dmc_termination = original
            def termination(physics):
                return None if self._suppress_old_goal_end else original(physics)
            self.raw.dmc_env.task.get_termination = termination
        try:
            return self._episode(case, protocol)
        finally:
            if original is not None:
                self.raw.dmc_env.task.get_termination = original

    def _episode(self, case, protocol):
        switch = protocol.track == "goal_switch"
        budget = protocol.switch_budget_steps if switch else protocol.deadline_steps
        pending = None
        queue = []
        plans, executed = [], []
        goal_version = 0
        switched = False
        hit = None
        old_goal_hits = []
        missed_feedback = 0
        waiting_steps = 0
        for tick in range(budget):
            if switch and tick == protocol.switch_step:
                self.change_goal()
                self._suppress_old_goal_end = False
                switched, goal_version = True, 1
                queue = []
                if pending is not None:
                    plans[pending[3]]["discarded_obsolete_goal"] = True
                    pending = None
            if pending is not None and plan_ready(tick, self.dt, pending[0]):
                queue = list(pending[1]) if pending[2] == goal_version else []
                pending = None
            if tick % protocol.feedback_steps == 0:
                if pending is not None:
                    missed_feedback += 1
                else:
                    actions, latency = self.make_plan(self.world.infos, self.history)
                    plans.append(dict(observation_step=tick, goal_version=goal_version,
                                      measured_seconds=latency, actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                                      available_at_seconds=tick * self.dt + (latency * protocol.latency_scale if switch else 0.0)))
                    if switch and latency * protocol.latency_scale > 0:
                        pending = (plans[-1]["available_at_seconds"], actions, goal_version, len(plans) - 1)
                        # The previous plan can be consumed only if any actions remain.
                    else:
                        queue = list(actions)
            if queue:
                action = queue.pop(0)
                source = "plan"
            else:
                action = self.last_action.copy()
                source = "hold"
                waiting_steps += 1
            term, trunc = self.step(action)
            executed.append(dict(step=tick + 1, source=source, action=action.tolist(), goal_version=goal_version))
            if term:
                if not switch or switched:
                    hit = tick + 1
                    break
                old_goal_hits.append(tick + 1)
            elif switch and not switched and self.task == "reacher" and \
                    self._native_dmc_termination(self.raw.dmc_env.physics) is not None:
                old_goal_hits.append(tick + 1)
            if trunc:
                raise RuntimeError("Unexpected simulator time limit before benchmark budget")
        compute_seconds = sum(p["measured_seconds"] for p in plans)
        elapsed_seconds = (hit if hit is not None else budget) * self.dt
        if not switch:
            elapsed_seconds += compute_seconds
        result = dict(case=case, task=self.task, arm=self.arm, track=protocol.track,
                      latency_scale=protocol.latency_scale if switch else None,
                      success=hit is not None, first_hit_step=hit, executed_steps=len(executed),
                      control_dt=self.dt, dt_source=self.dt_source, elapsed_seconds=elapsed_seconds,
                      cumulative_compute_seconds=compute_seconds, plans=plans, executed=executed,
                      old_goal_hit_steps=old_goal_hits, waiting_steps=waiting_steps,
                      missed_feedback=missed_feedback, switched=switched,
                      model_observation_steps=[p["observation_step"] for p in plans],
                      additional_training_steps=0)
        if not switch:
            result["deadline_curve"] = [{"factor": f, "seconds": f * 25 * self.dt,
                "success": deadline_success(hit is not None, hit, self.dt, compute_seconds, f * 25 * self.dt)}
                for f in DEADLINE_FACTORS]
        return result
