"""IK-based arm teleop controller.

Mink backend: MuJoCo differential IK QP solve against a MuJoCo XML model.

Requires: pip install mujoco-vr-teleop[ik]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class JointIKResult:
    joint_positions: np.ndarray


@dataclass
class JointIKConfig:
    xml_path: Path
    controlled_joint_names: list[str]
    target_frame_names: list[str]
    backend: str = "mink"
    home_joints: np.ndarray | None = None
    qpos0: np.ndarray | None = None
    smoothing_alpha: float = 0.95
    max_joint_delta: float = 0.12
    mink_solver: str = "daqp"
    mink_dt: float = 1.0 / 30.0
    mink_iterations: int = 4
    mink_gain: float = 0.20
    mink_max_velocity: float = 3.0
    mink_limit_gain: float = 0.5
    mink_limit_margin: float = 0.1
    mink_lm_damping: float = 1e-6
    mink_damping: float = 1e-6


# ---------------------------------------------------------------------------
# Joint arm + hand IK solver
# ---------------------------------------------------------------------------


class _JointIKBackend:
    backend_name = "base"
    backend_devices: list[str] = []

    def __init__(self, cfg: JointIKConfig):
        self.cfg = cfg
        self.num_joints = len(cfg.controlled_joint_names)
        if self.num_joints == 0:
            raise RuntimeError("Joint IK requires at least one controlled joint")
        self.home_joints = np.zeros(self.num_joints, dtype=float)
        if cfg.home_joints is not None:
            self.home_joints = np.asarray(cfg.home_joints, dtype=float).reshape(self.num_joints)
        self._smoothed: np.ndarray | None = None
        self._last_output: np.ndarray | None = None

    def info(self) -> dict:
        return {
            "backend": self.backend_name,
            "devices": self.backend_devices,
            "joint_names": list(self.cfg.controlled_joint_names),
        }

    def reset(self, home_joints: np.ndarray | None = None) -> None:
        if home_joints is not None:
            self.home_joints = self._sanitize_joints(home_joints)
        self._smoothed = None
        self._last_output = None

    def set_body_positions(self, body_positions: dict[str, np.ndarray]) -> None:
        return None

    def _sanitize_joints(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float).reshape(self.num_joints)

    def _limit_joint_delta(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
        max_delta = float(self.cfg.max_joint_delta)
        if max_delta <= 0.0 or not np.isfinite(max_delta):
            return q
        return reference + np.clip(q - reference, -max_delta, max_delta)

    def _run_ik(self, targets: list[dict], seed_q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def solve(self, targets: list[dict], seed_q: np.ndarray) -> JointIKResult:
        seed = self._sanitize_joints(seed_q)
        if self._smoothed is None:
            self._smoothed = seed.copy()
        raw = self._sanitize_joints(self._run_ik(targets, seed))
        raw = self._sanitize_joints(self._limit_joint_delta(raw, seed))
        output = self.cfg.smoothing_alpha * raw + (1.0 - self.cfg.smoothing_alpha) * self._smoothed
        reference = self._last_output if self._last_output is not None else seed
        output = self._sanitize_joints(self._limit_joint_delta(output, reference))
        self._smoothed = output.copy()
        self._last_output = output.copy()
        return JointIKResult(joint_positions=output)


class _MinkJointIKBackend(_JointIKBackend):
    backend_name = "mink-joint"

    def __init__(self, cfg: JointIKConfig):
        super().__init__(cfg)
        try:
            import mujoco
            import mink
        except ImportError as exc:
            raise SystemExit(
                "Missing Mink IK dependencies. Install with:\n"
                '  pip install "mujoco-vr-teleop[ik]"'
            ) from exc

        self._mink = mink
        self._mujoco = mujoco
        self.backend_devices = [f"solver={cfg.mink_solver}"]
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.configuration = mink.Configuration(self.model)
        self.q_template = (
            np.asarray(cfg.qpos0, dtype=float).reshape(self.model.nq).copy()
            if cfg.qpos0 is not None
            else np.zeros(self.model.nq, dtype=float)
        )

        self.qpos_adrs: list[int] = []
        self.dof_adrs: list[int] = []
        self._soft_lower = np.full(self.num_joints, -np.inf, dtype=float)
        self._soft_upper = np.full(self.num_joints, np.inf, dtype=float)
        for idx, joint_name in enumerate(cfg.controlled_joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise RuntimeError(f"Mink joint IK missing controlled joint: {joint_name}")
            self.qpos_adrs.append(int(self.model.jnt_qposadr[jid]))
            self.dof_adrs.append(int(self.model.jnt_dofadr[jid]))
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                margin = max(0.0, float(cfg.mink_limit_margin))
                if hi - lo > 2.0 * margin:
                    lo += margin
                    hi -= margin
                self._soft_lower[idx] = lo
                self._soft_upper[idx] = hi

        self.tasks_by_name = {}
        for frame_name in cfg.target_frame_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, frame_name) < 0:
                continue
            self.tasks_by_name[frame_name] = mink.FrameTask(
                frame_name=frame_name,
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.0,
                gain=float(cfg.mink_gain),
                lm_damping=float(cfg.mink_lm_damping),
            )
        if not self.tasks_by_name:
            raise RuntimeError("Mink joint IK has no target sites")

        self.limits = [
            mink.ConfigurationLimit(
                self.model,
                gain=float(cfg.mink_limit_gain),
                min_distance_from_limits=max(0.0, float(cfg.mink_limit_margin)),
            )
        ]
        if float(cfg.mink_max_velocity) > 0.0:
            self.limits.append(
                mink.VelocityLimit(
                    self.model,
                    {name: float(cfg.mink_max_velocity) for name in cfg.controlled_joint_names},
                )
            )
        self._controlled_dof_mask = np.zeros(self.model.nv, dtype=float)
        self._controlled_dof_mask[self.dof_adrs] = 1.0
        print(
            f"Joint IK solver ready ({self.backend_name}): {self.num_joints} controlled joints, "
            f"{len(self.tasks_by_name)} targets"
        )

    def set_body_positions(self, body_positions: dict[str, np.ndarray]) -> None:
        for body_name, pos in body_positions.items():
            bid = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                str(body_name),
            )
            if bid >= 0:
                self.model.body_pos[bid] = np.asarray(pos, dtype=float).reshape(3)
        self.configuration.update(q=self.configuration.q)

    def _sanitize_joints(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.num_joints)
        return np.clip(q, self._soft_lower, self._soft_upper)

    def _full_q(self, controlled_q: np.ndarray) -> np.ndarray:
        q = self.q_template.copy()
        for qadr, value in zip(self.qpos_adrs, self._sanitize_joints(controlled_q)):
            q[qadr] = value
        return q

    def _extract_q(self) -> np.ndarray:
        return self._sanitize_joints(
            np.array([self.configuration.q[qadr] for qadr in self.qpos_adrs], dtype=float)
        )

    def _run_ik(self, targets: list[dict], seed_q: np.ndarray) -> np.ndarray:
        self.configuration.update(q=self._full_q(seed_q))
        active_tasks = []
        for target in targets:
            task = self.tasks_by_name.get(str(target["frame_name"]))
            if task is None:
                continue
            task.set_position_cost(float(target.get("position_weight", 1.0)))
            task.set_orientation_cost(float(target.get("orientation_weight", 0.0)))
            pose = self._mink.SE3.from_rotation_and_translation(
                rotation=self._mink.SO3(np.asarray(target["quaternion"], dtype=float).reshape(4)),
                translation=np.asarray(target["position"], dtype=float).reshape(3),
            )
            task.set_target(pose)
            active_tasks.append(task)

        if not active_tasks:
            return seed_q.copy()

        try:
            iterations = max(1, int(self.cfg.mink_iterations))
            iteration_dt = float(self.cfg.mink_dt) / iterations
            for _ in range(iterations):
                vel = self._mink.solve_ik(
                    self.configuration,
                    active_tasks,
                    iteration_dt,
                    self.cfg.mink_solver,
                    damping=float(self.cfg.mink_damping),
                    limits=self.limits,
                )
                vel = np.asarray(vel, dtype=float) * self._controlled_dof_mask
                self.configuration.integrate_inplace(vel, iteration_dt)
        except self._mink.NoSolutionFound:
            self.configuration.update(q=self._full_q(seed_q))
            return seed_q.copy()

        return self._extract_q()


_JOINT_BACKENDS = {
    "mink": _MinkJointIKBackend,
}


class JointIKSolver:
    def __init__(self, cfg: JointIKConfig):
        backend = cfg.backend.lower()
        if backend not in _JOINT_BACKENDS:
            raise RuntimeError(
                f"Unknown joint IK backend '{cfg.backend}'. "
                f"Hand joint IK currently supports: {sorted(_JOINT_BACKENDS)}"
            )
        self._solver = _JOINT_BACKENDS[backend](cfg)
        self.cfg = cfg
        self.num_joints = self._solver.num_joints

    def info(self) -> dict:
        return self._solver.info()

    def reset(self, home_joints: np.ndarray | None = None) -> None:
        self._solver.reset(home_joints)

    def set_body_positions(self, body_positions: dict[str, np.ndarray]) -> None:
        self._solver.set_body_positions(body_positions)

    def solve(self, targets: list[dict], seed_q: np.ndarray) -> JointIKResult:
        return self._solver.solve(targets, seed_q)


