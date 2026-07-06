from __future__ import annotations

import copy
import re

import attrs
import mujoco
import numpy as np

from mujoco_vr_teleop import jenga_domain_randomization


DEFAULT_SETTLE_SECONDS = 0.5


def _body_subtree_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    """All geoms whose body is body_id or any descendant (transitive children).

    Used by the ``variant_select`` op so that toggling a slot's
    contype/conaffinity affects every geom attached underneath the slot's
    freejoint, including those inside the attached variant subtree.
    """
    descendants = {body_id}
    # Fixed-point: keep folding in bodies whose parent is already in the set.
    # nbody is small enough (~hundreds) that the O(n²) walk is fine.
    changed = True
    while changed:
        changed = False
        for bid in range(model.nbody):
            if bid in descendants:
                continue
            if int(model.body_parentid[bid]) in descendants:
                descendants.add(bid)
                changed = True
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants]


def _qmul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


@attrs.define(frozen=True)
class DomainRandomizer:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: dict
    enabled: bool
    tree: dict
    rng: np.random.Generator
    body_scale: np.ndarray
    body_visual_pos_offset: np.ndarray
    stream_body_ids: set[int]
    body_pos0: np.ndarray
    geom_pos0: np.ndarray
    geom_xpos0: np.ndarray
    geom_size0: np.ndarray
    geom_friction0: np.ndarray
    geom_contype0: np.ndarray
    geom_conaffinity0: np.ndarray
    mesh_vert0: np.ndarray
    qpos0: np.ndarray
    qvel0: np.ndarray
    mocap_pos0: np.ndarray | None
    mocap_quat0: np.ndarray | None
    sample_log: list[dict]

    @staticmethod
    def create(model: mujoco.MjModel, data: mujoco.MjData, config: dict | None) -> "DomainRandomizer":
        config = config if isinstance(config, dict) else {}
        tree = config.get("tree") if isinstance(config.get("tree"), dict) else {}
        enabled = bool(config.get("enabled", False))
        return DomainRandomizer(
            model=model,
            data=data,
            config=config,
            enabled=enabled,
            tree=tree,
            rng=np.random.default_rng(),
            body_scale=np.ones((model.nbody, 3), dtype=float),
            body_visual_pos_offset=np.zeros((model.nbody, 3), dtype=float),
            stream_body_ids=(
                DomainRandomizer._collect_stream_body_ids_for_config(model, config)
                if enabled
                else set()
            ),
            body_pos0=model.body_pos.copy(),
            geom_pos0=model.geom_pos.copy(),
            geom_xpos0=data.geom_xpos.copy(),
            geom_size0=model.geom_size.copy(),
            geom_friction0=model.geom_friction.copy(),
            geom_contype0=model.geom_contype.copy(),
            geom_conaffinity0=model.geom_conaffinity.copy(),
            mesh_vert0=model.mesh_vert.copy(),
            qpos0=data.qpos.copy(),
            qvel0=data.qvel.copy(),
            mocap_pos0=data.mocap_pos.copy() if model.nmocap > 0 else None,
            mocap_quat0=data.mocap_quat.copy() if model.nmocap > 0 else None,
            sample_log=[],
        )

    @staticmethod
    def _resolve_bodies_for_model(model: mujoco.MjModel, target: dict | None) -> list[int]:
        target = target or {}
        body_ids: list[int] = []
        for name in target.get("bodies", []):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
            if bid >= 0:
                body_ids.append(bid)
        pattern = target.get("bodies_regex") or target.get("body_regex")
        if pattern:
            regex = re.compile(str(pattern))
            for bid in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if regex.search(name):
                    body_ids.append(bid)
        return sorted(set(body_ids), key=body_ids.index)

    @staticmethod
    def _collect_stream_body_ids_for_model(model: mujoco.MjModel, node: dict | None) -> set[int]:
        if not isinstance(node, dict):
            return set()
        body_ids: set[int] = set()
        for action in node.get("actions", []):
            if not isinstance(action, dict):
                continue
            if action.get("op") in {
                "offset",
                "scale",
                "size_offset",
                "freejoint_pose",
                "scatter",
                "jenga_scatter",
            }:
                body_ids.update(
                    DomainRandomizer._resolve_bodies_for_model(model, action.get("target"))
                )
                for group in action.get("groups", []):
                    if isinstance(group, dict):
                        body_ids.update(
                            DomainRandomizer._resolve_bodies_for_model(
                                model, group.get("target", group)
                            )
                        )
                for gid in DomainRandomizer._resolve_geoms_for_model(model, action.get("target")):
                    body_ids.add(int(model.geom_bodyid[gid]))
        for child in node.get("children", []):
            body_ids.update(DomainRandomizer._collect_stream_body_ids_for_model(model, child))
        return body_ids

    @staticmethod
    def _collect_stream_body_ids_for_config(model: mujoco.MjModel, config: dict) -> set[int]:
        body_ids = DomainRandomizer._collect_stream_body_ids_for_model(model, config.get("tree"))
        modes = config.get("start_modes")
        if isinstance(modes, dict):
            for mode_config in modes.values():
                if isinstance(mode_config, dict):
                    body_ids.update(
                        DomainRandomizer._collect_stream_body_ids_for_model(
                            model,
                            mode_config.get("tree"),
                        )
                    )
        # Variant-pool slot bodies are addressed by name from the pool metadata,
        # not by DR target regex — collect them explicitly so the streamer
        # publishes their transforms (parked variants move under-table on swap;
        # the client renders that as them disappearing).
        pools = config.get("variant_pools")
        if isinstance(pools, dict):
            for pool in pools.values():
                if not isinstance(pool, dict):
                    continue
                for variant in pool.get("variants") or []:
                    name = variant.get("slot_body") if isinstance(variant, dict) else None
                    if not name:
                        continue
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                    if bid >= 0:
                        body_ids.add(bid)
        return body_ids

    @staticmethod
    def _resolve_geoms_for_model(model: mujoco.MjModel, target: dict | None) -> list[int]:
        target = target or {}
        geom_ids: list[int] = []
        for name in target.get("geoms", []):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
            if gid >= 0:
                geom_ids.append(gid)
        pattern = target.get("geoms_regex") or target.get("geom_regex")
        if pattern:
            regex = re.compile(str(pattern))
            for gid in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if regex.search(name):
                    geom_ids.append(gid)
        return sorted(set(geom_ids), key=geom_ids.index)

    def restore_baseline(self) -> None:
        self.model.body_pos[:] = self.body_pos0
        self.model.geom_pos[:] = self.geom_pos0
        self.model.geom_size[:] = self.geom_size0
        self.model.geom_friction[:] = self.geom_friction0
        self.model.geom_contype[:] = self.geom_contype0
        self.model.geom_conaffinity[:] = self.geom_conaffinity0
        self.model.mesh_vert[:] = self.mesh_vert0
        self.data.qpos[:] = self.qpos0
        self.data.qvel[:] = self.qvel0
        self.data.ctrl[:] = 0.0
        if self.mocap_pos0 is not None:
            self.data.mocap_pos[:] = self.mocap_pos0
        if self.mocap_quat0 is not None:
            self.data.mocap_quat[:] = self.mocap_quat0
        self.body_scale[:] = 1.0
        self.body_visual_pos_offset[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def active_start_mode(self) -> str | None:
        for sample in reversed(self.sample_log):
            if sample.get("op") == "start_mode":
                value = sample.get("value", {})
                return str(value.get("mode")) if value.get("mode") is not None else None
        return None

    def _sample_start_mode_config(self, forced_mode: str | None = None) -> dict:
        modes = self.config.get("start_modes")
        if not isinstance(modes, dict):
            return {}
        names = [str(name) for name, mode_config in modes.items() if isinstance(mode_config, dict)]
        if not names:
            return {}
        if forced_mode is not None:
            if forced_mode not in names:
                raise ValueError(
                    f"forced start mode {forced_mode!r} not in available modes {names}"
                )
            mode = forced_mode
            normalized_probability = 1.0
            sample_label = "forced"
        else:
            weights = np.array(
                [max(0.0, float(modes[name].get("probability", 1.0))) for name in names]
            )
            if float(weights.sum()) <= 0.0:
                raise ValueError("Start mode probabilities must sum to a positive value")
            mode = str(self.rng.choice(names, p=weights / weights.sum()))
            normalized_probability = float(weights[names.index(mode)] / weights.sum())
            sample_label = "weighted"
        mode_config = modes[mode]
        self.sample_log.append(
            {
                "name": "start_mode",
                "op": "start_mode",
                "sample_mode": sample_label,
                "target": {},
                "value": {
                    "mode": mode,
                    "probability": normalized_probability,
                },
            }
        )
        return mode_config

    def randomize(self, *, forced_mode: str | None = None) -> None:
        """Reset and re-randomize the scene.

        If ``forced_mode`` is given, pick that DR ``start_mode`` instead of
        sampling probabilistically. Use this when a task constrains which
        mode is valid for it.
        """
        self.restore_baseline()
        self.sample_log.clear()
        if self.enabled:
            active_config = self._sample_start_mode_config(forced_mode=forced_mode)
            self._apply_node(self.tree)
            if active_config:
                self._apply_node(active_config.get("tree"))
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def randomize_and_settle(
        self,
        *,
        max_attempts: int = 1,
        check_quiescence: dict | None = None,
        forced_mode: str | None = None,
    ) -> int:
        """Run ``randomize`` + ``settle`` with optional rejection sampling.

        If ``max_attempts > 1`` and ``check_quiescence`` is provided, we retry
        the whole randomize-then-settle cycle whenever the result has any
        target free body still moving above the configured thresholds. This is
        the foolproof wrapper for "I want a clean reset every time" — the
        underlying scene sampler may occasionally produce wobbly poses but
        re-sampling shakes them off.

        ``forced_mode`` (if not ``None``) pins the DR start mode, bypassing
        the probabilistic sample. Use it when the upcoming task constrains
        which mode is valid.

        Returns the number of attempts used (1-indexed).
        """
        for attempt in range(1, max_attempts + 1):
            self.randomize(forced_mode=forced_mode)
            self.settle()
            if check_quiescence is None:
                return attempt
            if self._is_quiescent(check_quiescence):
                return attempt
        return max_attempts

    def _is_quiescent(self, config: dict) -> bool:
        """Check whether all matching free bodies are moving below thresholds."""
        pattern = config.get("bodies_regex")
        max_lin = float(config.get("max_linear_speed", 0.3))
        max_ang = float(config.get("max_angular_speed", 1.0))
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if pattern and not re.fullmatch(pattern, name):
                continue
            dadr = self._body_freejoint_dadr(bid)
            if dadr is None:
                continue
            v_lin = float(np.linalg.norm(self.data.qvel[dadr : dadr + 3]))
            v_ang = float(np.linalg.norm(self.data.qvel[dadr + 3 : dadr + 6]))
            if v_lin > max_lin or v_ang > max_ang:
                return False
        return True

    def settle(self) -> None:
        config_source = self.config
        modes = self.config.get("start_modes")
        mode = self.active_start_mode()
        if isinstance(modes, dict) and mode in modes and isinstance(modes[mode], dict):
            config_source = modes[mode]
        config = config_source.get("settle", self.config.get("settle", {}))
        if config is False or config is None:
            return
        if not isinstance(config, dict):
            raise ValueError(f"Expected domain randomization settle config dict, got: {config}")
        seconds = float(config.get("seconds", DEFAULT_SETTLE_SECONDS))
        passes = config.get("passes", [])
        if isinstance(passes, list) and passes:
            for settle_pass in passes:
                if not isinstance(settle_pass, dict):
                    continue
                if settle_pass.get("op") == "assemble":
                    self._settle_assemble(settle_pass, seconds)
                else:
                    raise ValueError(f"Unsupported domain randomization settle op: {settle_pass.get('op')}")
        elif seconds > 0.0:
            self._step_settle(seconds)

        # Optional: extend the settle adaptively until free bodies are quiet
        # enough. Useful for scenes whose objects wobble (curved-bottom mugs,
        # plates with thin rims) — a fixed-time settle can leave them moving.
        extend = config.get("extend_until_stable")
        if isinstance(extend, dict):
            self._settle_until_stable(extend)

    def _settle_until_stable(self, config: dict) -> None:
        """Step the sim in chunks until free-body velocities are below a
        threshold, or we exhaust the extension budget. Targets only the free
        bodies whose names match the optional ``bodies_regex``.
        """
        max_total_seconds = float(config.get("max_extra_seconds", 4.0))
        chunk_seconds = float(config.get("chunk_seconds", 0.25))
        max_lin = float(config.get("max_linear_speed", 0.2))
        max_ang = float(config.get("max_angular_speed", 0.5))
        pattern = config.get("bodies_regex")
        body_ids: list[int] = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if pattern and not re.fullmatch(pattern, name):
                continue
            dadr = self._body_freejoint_dadr(bid)
            if dadr is None:
                continue
            body_ids.append(bid)
        if not body_ids:
            return
        dadrs = [self._body_freejoint_dadr(bid) for bid in body_ids]
        elapsed = 0.0
        while elapsed < max_total_seconds:
            quiet = True
            for dadr in dadrs:
                v_lin = float(np.linalg.norm(self.data.qvel[dadr : dadr + 3]))
                v_ang = float(np.linalg.norm(self.data.qvel[dadr + 3 : dadr + 6]))
                if v_lin > max_lin or v_ang > max_ang:
                    quiet = False
                    break
            if quiet:
                return
            self._step_settle(chunk_seconds)
            elapsed += chunk_seconds

    def _step_settle(self, seconds: float) -> None:
        for _ in range(max(0, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def _settle_assemble(self, settle_pass: dict, default_seconds: float) -> None:
        groups = self._resolve_body_groups(settle_pass, freejoint_only=True)
        if not groups:
            return
        body_ids = sorted({bid for group in groups for bid in group})
        qadr_by_body = {bid: self._body_freejoint_qadr(bid) for bid in body_ids}
        qpos_by_body = {
            bid: self.data.qpos[qadr : qadr + 7].copy()
            for bid, qadr in qadr_by_body.items()
            if qadr is not None
        }
        hide_offset = self._sample_vec3(settle_pass.get("hide_offset", [0.0, 0.0, 2.0]))
        drop_height = float(settle_pass.get("drop_height", 0.0))
        drop_pitch_spec = settle_pass.get("drop_pitch_deg")
        seconds = float(settle_pass.get("seconds", default_seconds))
        hidden = set(body_ids)
        drop_samples = []
        for group in groups:
            hidden.difference_update(group)
            for bid in hidden:
                qadr = qadr_by_body[bid]
                target = qpos_by_body[bid]
                self.data.qpos[qadr : qadr + 3] = target[:3] + hide_offset
                self.data.qpos[qadr + 3 : qadr + 7] = target[3:]
            for bid in group:
                qadr = qadr_by_body[bid]
                target = qpos_by_body[bid]
                quat = target[3:7].copy()
                pitch_deg = 0.0
                if drop_pitch_spec is not None:
                    pitch_deg = self._sample_scalar(drop_pitch_spec)
                    quat = _qmul_wxyz(quat, self._quat_from_euler_deg([0.0, pitch_deg, 0.0]))
                    quat /= np.linalg.norm(quat)
                self.data.qpos[qadr : qadr + 3] = target[:3] + [0.0, 0.0, drop_height]
                self.data.qpos[qadr + 3 : qadr + 7] = quat
                if drop_height or drop_pitch_spec is not None:
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
                    drop_samples.append(
                        {
                            "body_name": name,
                            "drop_height": drop_height,
                            "drop_pitch_deg": pitch_deg,
                        }
                    )
            self._step_settle(seconds)
        if drop_samples:
            self.sample_log.append(
                {
                    "name": settle_pass.get("name"),
                    "op": "settle_assemble",
                    "sample_mode": "per_target",
                    "target": copy.deepcopy(settle_pass.get("target", {})),
                    "value": {"drops": drop_samples},
                }
            )

    def metadata(self) -> dict:
        body_scales = {}
        for bid, scale in enumerate(self.body_scale):
            scale = np.asarray(scale, dtype=float)
            if np.allclose(scale, 1.0, atol=1e-12, rtol=0.0):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
            body_scales[name] = scale.tolist()
        return {
            "enabled": bool(self.enabled),
            "start_mode": self.active_start_mode(),
            "samples": copy.deepcopy(self.sample_log),
            "body_scales": body_scales,
        }

    def _sample_scalar(self, spec) -> float:
        if isinstance(spec, dict):
            if "uniform" in spec:
                lo, hi = spec["uniform"]
                value = float(self.rng.uniform(float(lo), float(hi)))
                if "step" in spec:
                    step = float(spec["step"])
                    if step > 0.0:
                        value = round(value / step) * step
                        value = min(max(value, float(lo)), float(hi))
                return value
            raise ValueError(f"Unsupported random distribution: {spec}")
        return float(spec)

    def _sample_vec3(self, spec) -> np.ndarray:
        if spec is None:
            return np.zeros(3, dtype=float)
        values = list(spec)
        if len(values) != 3:
            raise ValueError(f"Expected 3-vector randomization value, got: {spec}")
        return np.array([self._sample_scalar(v) for v in values], dtype=float)

    def _quat_from_euler_deg(self, euler_deg) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)

    def _log_sample(self, action: dict, value: dict) -> None:
        self.sample_log.append(
            {
                "name": action.get("name"),
                "op": action.get("op"),
                "sample_mode": action.get("sample", "shared"),
                "target": copy.deepcopy(action.get("target", {})),
                "value": value,
            }
        )

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _body_names(self, body_ids: list[int]) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
            for bid in body_ids
        ]

    def _geom_names(self, geom_ids: list[int]) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
            for gid in geom_ids
        ]

    def _resolve_bodies(self, target: dict | None) -> list[int]:
        target = target or {}
        body_ids: list[int] = []
        for name in target.get("bodies", []):
            bid = self._body_id(str(name))
            if bid >= 0:
                body_ids.append(bid)
        pattern = target.get("bodies_regex") or target.get("body_regex")
        if pattern:
            regex = re.compile(str(pattern))
            for bid in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                if regex.search(name):
                    body_ids.append(bid)
        return sorted(set(body_ids), key=body_ids.index)

    def _resolve_geoms(self, target: dict | None) -> list[int]:
        target = target or {}
        geom_ids: list[int] = []
        for name in target.get("geoms", []):
            gid = self._geom_id(str(name))
            if gid >= 0:
                geom_ids.append(gid)
        pattern = target.get("geoms_regex") or target.get("geom_regex")
        if pattern:
            regex = re.compile(str(pattern))
            for gid in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if regex.search(name):
                    geom_ids.append(gid)
        return sorted(set(geom_ids), key=geom_ids.index)

    def _body_freejoint_qadr(self, body_id: int) -> int | None:
        jadr = int(self.model.body_jntadr[body_id])
        jnum = int(self.model.body_jntnum[body_id])
        for jid in range(jadr, jadr + jnum):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[jid])
        return None

    def _body_freejoint_dadr(self, body_id: int) -> int | None:
        jadr = int(self.model.body_jntadr[body_id])
        jnum = int(self.model.body_jntnum[body_id])
        for jid in range(jadr, jadr + jnum):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_dofadr[jid])
        return None

    def _resolve_body_groups(self, action: dict, *, freejoint_only: bool = False) -> list[list[int]]:
        groups_spec = action.get("groups")
        if isinstance(groups_spec, list) and groups_spec:
            groups = [
                self._resolve_bodies(group.get("target", group) if isinstance(group, dict) else None)
                for group in groups_spec
            ]
        else:
            groups = [self._resolve_bodies(action.get("target"))]
        if freejoint_only:
            groups = [
                [bid for bid in group if self._body_freejoint_qadr(bid) is not None]
                for group in groups
            ]
        return [group for group in groups if group]


    def _apply_node(self, node: dict | None) -> None:
        if not isinstance(node, dict):
            return
        snapshot = {
            "body_pos": self.model.body_pos.copy(),
            "geom_pos": self.model.geom_pos.copy(),
            "geom_xpos": self.data.geom_xpos.copy(),
            "geom_size": self.model.geom_size.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "mesh_vert": self.model.mesh_vert.copy(),
            "qpos": self.data.qpos.copy(),
            "mocap_pos": None if self.model.nmocap == 0 else self.data.mocap_pos.copy(),
            "body_scale": self.body_scale.copy(),
        }
        for action in node.get("actions", []):
            if not isinstance(action, dict):
                continue
            op = action.get("op")
            if op == "offset":
                self._apply_offset(action, snapshot)
            elif op == "scale":
                self._apply_scale(action, snapshot)
            elif op == "size_offset":
                self._apply_size_offset(action, snapshot)
            elif op == "freejoint_pose":
                self._apply_freejoint_pose(action, snapshot)
            elif op == "sequential_drop":
                self._apply_sequential_drop(action, snapshot)
            elif op == "grid_place":
                self._apply_grid_place(action, snapshot)
            elif op in {"scatter", "jenga_scatter"}:
                jenga_domain_randomization.apply_scatter(self, action, snapshot)
            elif op == "scatter_drop":
                self._apply_scatter_drop(action, snapshot)
            else:
                raise ValueError(f"Unsupported domain randomization op: {op}")
        mujoco.mj_forward(self.model, self.data)
        for child in node.get("children", []):
            self._apply_node(child)

    def _apply_offset(self, action: dict, snapshot: dict) -> None:
        body_ids = self._resolve_bodies(action.get("target"))
        geom_ids = self._resolve_geoms(action.get("target"))
        if not body_ids and not geom_ids:
            return
        sample = action.get("sample", "shared")
        if sample == "per_target":
            for bid in body_ids:
                offset = self._sample_vec3(action.get("pos"))
                actual_offset = self._apply_body_offset(bid, offset, snapshot)
                self._log_sample(
                    action,
                    {"pos": actual_offset.tolist(), "body_names": self._body_names([bid])},
                )
            for gid in geom_ids:
                offset = self._sample_vec3(action.get("pos"))
                pos = snapshot["geom_pos"][gid] + offset
                actual_offset = pos - snapshot["geom_pos"][gid]
                self.model.geom_pos[gid] = pos
                self.body_visual_pos_offset[int(self.model.geom_bodyid[gid])] = actual_offset
                self._log_sample(
                    action,
                    {"pos": actual_offset.tolist(), "geom_names": self._geom_names([gid])},
                )
            return
        offset = self._sample_vec3(action.get("pos"))
        for bid in body_ids:
            self._apply_body_offset(bid, offset, snapshot)
        for gid in geom_ids:
            self.model.geom_pos[gid] = snapshot["geom_pos"][gid] + offset
        for bid in sorted({int(self.model.geom_bodyid[gid]) for gid in geom_ids}):
            gid = next(g for g in geom_ids if int(self.model.geom_bodyid[g]) == bid)
            self.body_visual_pos_offset[bid] = self.model.geom_pos[gid] - snapshot["geom_pos"][gid]
        self._log_sample(
            action,
            {
                "pos": offset.tolist(),
                "body_names": self._body_names(body_ids),
                "geom_names": self._geom_names(geom_ids),
            },
        )

    def _apply_body_offset(self, body_id: int, offset: np.ndarray, snapshot: dict) -> np.ndarray:
        pos = snapshot["body_pos"][body_id] + offset
        actual_offset = pos - snapshot["body_pos"][body_id]
        self.model.body_pos[body_id] = pos
        return actual_offset

    def _sample_scale_factor(self, spec) -> np.ndarray:
        # If a single scalar/dist is provided we treat it as isotropic — one
        # draw, reused across all three axes. Pass a 3-element list to get
        # per-axis sampling.
        if not isinstance(spec, (list, tuple)):
            value = self._sample_scalar(spec)
            return np.array([value, value, value], dtype=float)
        if len(spec) != 3:
            raise ValueError(f"Expected 3 scale factors, got: {spec}")
        return np.array([self._sample_scalar(axis_spec) for axis_spec in spec], dtype=float)

    def _scale_reference(self, body_ids: list[int], snapshot: dict, *, per_axis: bool) -> np.ndarray | None:
        references: list[np.ndarray] = []
        for bid in body_ids:
            for gid in np.flatnonzero(self.model.geom_bodyid == bid):
                size = np.asarray(snapshot["geom_size"][gid], dtype=float)
                if size.size < 3:
                    continue
                if per_axis:
                    references.append(2.0 * size[:3])
                else:
                    references.append(np.full(3, 2.0 * float(np.max(size))))
        if not references:
            return None
        return np.max(np.vstack(references), axis=0)

    def _apply_scale(self, action: dict, snapshot: dict) -> None:
        body_ids = self._resolve_bodies(action.get("target"))
        if not body_ids:
            return
        sample = action.get("sample", "shared")
        factor_spec = action.get("factor", 1.0)
        shared_factor = self._sample_scale_factor(factor_spec)
        factors: dict[int, np.ndarray] = {}
        scaled_meshes: set[int] = set()
        for bid in body_ids:
            factor = self._sample_scale_factor(factor_spec) if sample == "per_target" else shared_factor
            factors[bid] = factor
            self.body_scale[bid] = snapshot["body_scale"][bid] * factor
            geom_ids = np.flatnonzero(self.model.geom_bodyid == bid)
            for gid in geom_ids:
                self.model.geom_size[gid] = snapshot["geom_size"][gid] * factor
                if self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = int(self.model.geom_dataid[gid])
                if mesh_id < 0 or mesh_id in scaled_meshes:
                    continue
                vadr = int(self.model.mesh_vertadr[mesh_id])
                vnum = int(self.model.mesh_vertnum[mesh_id])
                self.model.mesh_vert[vadr : vadr + vnum] = snapshot["mesh_vert"][vadr : vadr + vnum] * factor
                scaled_meshes.add(mesh_id)
        origins = self._apply_scale_pose(action, body_ids, factors, snapshot)
        log_value = {
            "factor": shared_factor.tolist(),
            "factors_by_body": {
                name: factors[bid].tolist()
                for bid, name in zip(body_ids, self._body_names(body_ids))
            },
            "body_names": self._body_names(body_ids),
        }
        if origins is not None:
            origin, source_origin = origins
            log_value["origin"] = origin.tolist()
            log_value["source_origin"] = source_origin.tolist()
        self._log_sample(action, log_value)

    def _apply_size_offset(self, action: dict, snapshot: dict) -> None:
        body_ids = self._resolve_bodies(action.get("target"))
        if not body_ids:
            return
        sample = action.get("sample", "shared")
        delta_spec = action.get("delta", action.get("size", 0.0))
        shared_delta = self._sample_scalar(delta_spec)
        deltas: dict[int, float] = {}
        factors: dict[int, float] = {}
        for bid in body_ids:
            delta = self._sample_scalar(delta_spec) if sample == "per_target" else shared_delta
            reference_vec = self._scale_reference([bid], snapshot, per_axis=False)
            reference = None if reference_vec is None else float(reference_vec[0])
            if reference is None or reference <= 0.0:
                continue
            target_reference = reference + delta
            factor = max(float(target_reference / reference), 1e-9)
            deltas[bid] = float(target_reference - reference)
            factors[bid] = factor
            self.body_scale[bid] = snapshot["body_scale"][bid] * factor
            for gid in np.flatnonzero(self.model.geom_bodyid == bid):
                self.model.geom_size[gid] = snapshot["geom_size"][gid] * factor
        self._log_sample(
            action,
            {
                "delta_by_body": {
                    name: deltas[bid] for bid, name in zip(body_ids, self._body_names(body_ids))
                    if bid in deltas
                },
                "factor_by_body": {
                    name: factors[bid] for bid, name in zip(body_ids, self._body_names(body_ids))
                    if bid in factors
                },
                "body_names": self._body_names(body_ids),
            },
        )

    def _apply_scale_pose(
        self,
        action: dict,
        body_ids: list[int],
        factors: dict[int, np.ndarray],
        snapshot: dict,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        pose = action.get("pose")
        if not isinstance(pose, dict):
            return None
        mode = pose.get("mode")
        if mode != "freejoint_positions":
            raise ValueError(f"Unsupported scale pose mode: {mode}")
        qaddrs_by_body = {
            bid: qadr for bid in body_ids if (qadr := self._body_freejoint_qadr(bid)) is not None
        }
        if not qaddrs_by_body:
            return None
        origin, source_origin = self._resolve_scale_origins(
            pose.get("origin"),
            list(qaddrs_by_body.values()),
            snapshot,
        )
        for bid, qadr in qaddrs_by_body.items():
            xmat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(xmat, snapshot["qpos"][qadr + 3 : qadr + 7])
            xmat = xmat.reshape(3, 3)
            factor = np.abs(xmat) @ factors[bid]
            self.data.qpos[qadr : qadr + 3] = (
                origin + factor * (snapshot["qpos"][qadr : qadr + 3] - source_origin)
            )
        return origin, source_origin

    def _resolve_scale_origins(
        self,
        origin_spec,
        qaddrs: list[int],
        snapshot: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        origin_spec = origin_spec if isinstance(origin_spec, dict) else {}
        positions = np.array([snapshot["qpos"][qadr : qadr + 3] for qadr in qaddrs], dtype=float)
        center = positions.mean(axis=0)
        origin = center.copy()
        source_origin = center.copy()

        xy_spec = origin_spec.get("xy", "group_center")
        if xy_spec == "group_center":
            origin[:2] = center[:2]
            source_origin[:2] = center[:2]
        elif isinstance(xy_spec, (list, tuple)) and len(xy_spec) == 2:
            origin[:2] = [float(xy_spec[0]), float(xy_spec[1])]
            source_origin[:2] = origin[:2]
        else:
            raise ValueError(f"Unsupported scale origin xy: {xy_spec}")

        z_spec = origin_spec.get("z", "group_center")
        if z_spec == "group_center":
            origin[2] = center[2]
            source_origin[2] = center[2]
        elif isinstance(z_spec, dict) and "geom_top" in z_spec:
            gid = self._geom_id(str(z_spec["geom_top"]))
            if gid < 0:
                raise ValueError(f"Unknown geom for scale origin: {z_spec['geom_top']}")
            origin[2] = float(
                snapshot["geom_xpos"][gid, 2]
                + snapshot["geom_size"][gid, 2]
                + float(z_spec.get("clearance", 0.0))
            )
            source_origin[2] = float(self.geom_xpos0[gid, 2] + self.geom_size0[gid, 2])
        else:
            origin[2] = float(z_spec)
            source_origin[2] = origin[2]
        return origin, source_origin

    def _apply_freejoint_pose(self, action: dict, snapshot: dict) -> None:
        body_ids = [
            bid
            for bid in self._resolve_bodies(action.get("target"))
            if self._body_freejoint_qadr(bid) is not None
        ]
        if not body_ids:
            return
        sample = action.get("sample", "shared")
        if sample == "per_target":
            groups = [[bid] for bid in body_ids]
        elif sample == "per_group":
            groups = self._resolve_body_groups(action, freejoint_only=True)
        else:
            groups = [body_ids]
        for group in groups:
            qaddrs = [self._body_freejoint_qadr(bid) for bid in group]
            qaddrs = [qadr for qadr in qaddrs if qadr is not None]
            if not qaddrs:
                continue
            center = np.mean([snapshot["qpos"][qadr : qadr + 3] for qadr in qaddrs], axis=0)
            offset = self._sample_vec3(action.get("pos"))
            euler_deg = self._sample_vec3(action.get("euler_deg", [0.0, 0.0, 0.0]))
            delta_quat = self._quat_from_euler_deg(euler_deg)
            from scipy.spatial.transform import Rotation

            delta_rot = Rotation.from_quat(
                [delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0]]
            ).as_matrix()
            for qadr in qaddrs:
                base_pos = snapshot["qpos"][qadr : qadr + 3]
                base_quat = snapshot["qpos"][qadr + 3 : qadr + 7]
                if action.get("about") == "group_center":
                    pos = center + delta_rot @ (base_pos - center) + offset
                else:
                    pos = base_pos + offset
                quat = _qmul_wxyz(delta_quat, base_quat)
                quat /= np.linalg.norm(quat)
                self.data.qpos[qadr : qadr + 3] = pos
                self.data.qpos[qadr + 3 : qadr + 7] = quat
            self._log_sample(
                action,
                {
                    "pos": offset.tolist(),
                    "euler_deg": euler_deg.tolist(),
                    "body_names": self._body_names(group),
                },
            )

    def _apply_variant_select(self, action: dict, snapshot: dict) -> None:
        """Pick a variant from a pool and apply it to the slot's geoms.

        Thin variant_pools pattern: one slot body per kind, all variant
        meshes embedded in the asset table. This op picks a variant id
        (sampled or pinned via ``selected_variants``) and calls
        ``variant_pools.apply_variant`` which swaps geom_dataid (mesh slots)
        or geom_pos/size/quat (box slots) for the slot's geoms.

        Required action field:
          ``pool``: pool name registered in ``config["variant_pools"]``

        Initial item placement is handled separately by a ``scatter_drop``
        op (or a caller-provided ``scatter_bodies`` invocation) — this op
        only mutates geometry, not pose.
        """
        from mujoco_vr_teleop.variant_pools import apply_variant

        pool_name = action.get("pool")
        if not pool_name:
            raise ValueError("variant_select op requires a 'pool' field")
        pools = self.config.get("variant_pools") or {}
        pool = pools.get(pool_name)
        if not isinstance(pool, dict):
            raise ValueError(
                f"variant_select: pool {pool_name!r} not found in config['variant_pools']. "
                f"Known pools: {sorted(pools.keys())}"
            )
        ids = pool.get("ids") or []
        if not ids:
            return

        pinned_map = self.config.get("selected_variants") or {}
        pinned_id = pinned_map.get(pool_name) if isinstance(pinned_map, dict) else None
        if pinned_id is not None:
            if int(pinned_id) not in [int(v) for v in ids]:
                raise ValueError(
                    f"variant_select: pinned id {pinned_id} not in pool {pool_name} "
                    f"(have: {ids})"
                )
            chosen_id = int(pinned_id)
            sample_mode = "pinned"
        else:
            chosen_id = int(self.rng.choice(ids))
            sample_mode = "uniform"

        apply_variant(self.model, pool, chosen_id)
        mujoco.mj_forward(self.model, self.data)

        log_value: dict = {
            "pool": pool_name,
            "variant_id": chosen_id,
            "slot_body": pool.get("slot_body"),
            "sample_mode": sample_mode,
        }
        self._log_sample(action, log_value)

    def _apply_scatter_drop(self, action: dict, snapshot: dict) -> None:
        """Scatter every slot body (from variant_pools) onto a surface using
        scatter_reset.scatter_bodies. Required fields:
          ``region``: {x_lo, x_hi, y_lo, y_hi, surface_z, ...}
          ``priorities`` (optional): {pool_name: int} drop order
        """
        from mujoco_vr_teleop.scatter_reset import ScatterRegion, scatter_bodies

        # Resolve scatter targets from variant_pools.SCENE_POOLS: each pool's
        # slot body name is fixed (no per-vid suffix). The action's
        # `priorities` map keys by pool name.
        from mujoco_vr_teleop.variant_pools import SCENE_POOLS

        scene_type = action.get("scene_type")
        if not scene_type:
            scene_type = self.config.get("scene_type")
        # Fall back: scan every pool registered for any scene whose slot
        # body actually exists in this model.
        pools_iter = (
            SCENE_POOLS.get(scene_type, [])
            if scene_type
            else [p for ps in SCENE_POOLS.values() for p in ps]
        )
        body_ids = []
        body_priorities: dict[int, int] = {}
        priority_map = action.get("priorities") or {}
        for pool in pools_iter:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pool.slot_body)
            if bid < 0:
                continue
            body_ids.append(bid)
            if pool.name in priority_map:
                body_priorities[bid] = int(priority_map[pool.name])
        if not body_ids:
            return

        region_spec = action.get("region") or {}
        region = ScatterRegion(
            x_lo=float(region_spec.get("x_lo", 0.32)),
            x_hi=float(region_spec.get("x_hi", 0.96)),
            y_lo=float(region_spec.get("y_lo", -0.58)),
            y_hi=float(region_spec.get("y_hi", 0.58)),
            surface_z=float(region_spec.get("surface_z", 0.78)),
            obstacle_margin=float(region_spec.get("obstacle_margin", 0.08)),
            settle_between_s=float(region_spec.get("settle_between_s", 0.3)),
            final_settle_s=float(region_spec.get("final_settle_s", 0.5)),
        )
        result = scatter_bodies(self.model, self.data, self.rng,
                                body_ids, region,
                                priorities=body_priorities or None)
        self._log_sample(action, {"placements": result.get("placements", [])})

    def _apply_variant_subtree_scale(
        self,
        slot_bid: int,
        factor: np.ndarray,
        snapshot: dict,
    ) -> None:
        """Scale every geom/mesh under ``slot_bid`` (recursively) by ``factor``.

        Mirrors the body-scoped path inside ``_apply_scale`` but walks the full
        subtree so all bodies attached under the variant slot are scaled
        consistently. The slot's freejoint qpos is left untouched — caller is
        responsible for placing the slot at the target pose (which happens
        before this is called).
        """
        descendants: set[int] = {slot_bid}
        changed = True
        while changed:
            changed = False
            for bid in range(self.model.nbody):
                if bid in descendants:
                    continue
                if int(self.model.body_parentid[bid]) in descendants:
                    descendants.add(bid)
                    changed = True
        scaled_meshes: set[int] = set()
        for bid in descendants:
            self.body_scale[bid] = snapshot["body_scale"][bid] * factor
            for gid in np.flatnonzero(self.model.geom_bodyid == bid):
                self.model.geom_size[gid] = snapshot["geom_size"][gid] * factor
                if self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = int(self.model.geom_dataid[gid])
                if mesh_id < 0 or mesh_id in scaled_meshes:
                    continue
                vadr = int(self.model.mesh_vertadr[mesh_id])
                vnum = int(self.model.mesh_vertnum[mesh_id])
                self.model.mesh_vert[vadr : vadr + vnum] = (
                    snapshot["mesh_vert"][vadr : vadr + vnum] * factor
                )
                scaled_meshes.add(mesh_id)

    def _apply_sequential_drop(self, action: dict, snapshot: dict) -> None:
        """Drop target bodies onto the table one at a time.

        For each body we sample (x, y) in the configured region, rejecting any
        position that is within ``radius`` of an already-placed body's xy. That
        keeps drops from landing on top of each other, so each one settles
        cleanly without disturbing its neighbours. Bodies not yet placed are
        parked far below the table while waiting.

        Per-target ``radius`` comes from a ``radii`` map keyed by body-name
        regex; the default is taken from ``default_radius``. Set
        ``edge_margin`` to keep all sampled positions a bit inside the region
        bounds so an object that lands near the edge doesn't roll off.
        """
        body_ids = [
            bid
            for bid in self._resolve_bodies(action.get("target"))
            if self._body_freejoint_qadr(bid) is not None
        ]
        if not body_ids:
            return

        region = action.get("region", {})
        x_range = region.get("x", [0.4, 0.9])
        y_range = region.get("y", [-0.1, 0.5])
        drop_z = float(region.get("z", 0.85))
        avoid_rects = action.get("avoid", []) or []
        yaw_spec = action.get("yaw_deg", 0.0)
        roll_spec = action.get("roll_deg", 0.0)
        pitch_spec = action.get("pitch_deg", 0.0)
        settle_per_drop = float(action.get("settle_per_drop", 0.3))
        edge_margin = float(action.get("edge_margin", 0.0))
        default_radius = float(action.get("default_radius", 0.05))
        # Per-body radius lookup: action["radii"] is a list of
        # {"bodies_regex": ..., "radius": ...} entries in priority order.
        radii_specs = action.get("radii", []) or []
        # Where to park bodies before they're dropped — far below the table so
        # gravity doesn't pull them into anything during the settle phase.
        hide_offset = np.asarray(action.get("hide_offset", [0.0, 0.0, -5.0]), dtype=float)
        max_resample = int(action.get("max_resample", 200))

        qaddrs = {bid: self._body_freejoint_qadr(bid) for bid in body_ids}
        dadr_for = {bid: self._body_freejoint_dadr(bid) for bid in body_ids}

        def zero_velocity(bid: int) -> None:
            dadr = dadr_for[bid]
            if dadr is not None:
                self.data.qvel[dadr : dadr + 6] = 0.0

        def in_any_avoid(x: float, y: float) -> bool:
            for rect in avoid_rects:
                xr = rect.get("x", [0.0, 0.0])
                yr = rect.get("y", [0.0, 0.0])
                if xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]:
                    return True
            return False

        def radius_for(bid: int) -> float:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            for spec in radii_specs:
                pattern = spec.get("bodies_regex")
                if pattern and re.fullmatch(pattern, name):
                    return float(spec.get("radius", default_radius))
            return default_radius

        # Park everyone underground first so nothing intersects during placement.
        # Spread them out so the underground "park" itself has no overlap.
        for idx, bid in enumerate(body_ids):
            qadr = qaddrs[bid]
            park = np.array(
                [hide_offset[0] + 0.4 * (idx % 6), hide_offset[1] + 0.4 * (idx // 6), hide_offset[2]],
                dtype=float,
            )
            self.data.qpos[qadr : qadr + 3] = park
            self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            zero_velocity(bid)
        mujoco.mj_forward(self.model, self.data)

        samples = []
        # placed_xys is the xy of every body we've already dropped, paired
        # with its radius. We reject any new sample that puts the candidate
        # within sum_of_radii of an existing placement.
        placed_xys: list[tuple[float, float, float]] = []
        x_lo = x_range[0] + edge_margin
        x_hi = x_range[1] - edge_margin
        y_lo = y_range[0] + edge_margin
        y_hi = y_range[1] - edge_margin
        for bid in body_ids:
            qadr = qaddrs[bid]
            r = radius_for(bid)
            x = y = 0.0
            placed = False
            for _ in range(max_resample):
                x = float(self.rng.uniform(x_lo, x_hi))
                y = float(self.rng.uniform(y_lo, y_hi))
                if in_any_avoid(x, y):
                    continue
                if any(
                    (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2
                    for px, py, pr in placed_xys
                ):
                    continue
                placed = True
                break
            if not placed:
                # Couldn't find a clear spot — shouldn't happen with sensible
                # region sizes, but if it does, fall back to a spot just
                # outside any existing placement.
                x = 0.5 * (x_lo + x_hi)
                y = 0.5 * (y_lo + y_hi)
            yaw_deg = self._sample_scalar(yaw_spec)
            roll_deg = self._sample_scalar(roll_spec)
            pitch_deg = self._sample_scalar(pitch_spec)
            # Compose sampled rotation onto the body's baseline orientation so
            # e.g. a bottle that lays on its side at rest stays roughly on its
            # side, just yawed/wobbled.
            base_quat = np.asarray(snapshot["qpos"][qadr + 3 : qadr + 7], dtype=float)
            base_norm = np.linalg.norm(base_quat)
            if base_norm < 1e-9:
                base_quat = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                base_quat = base_quat / base_norm
            delta_quat = self._quat_from_euler_deg([roll_deg, pitch_deg, yaw_deg])
            quat = _qmul_wxyz(delta_quat, base_quat)
            quat = quat / np.linalg.norm(quat)

            self.data.qpos[qadr : qadr + 3] = [x, y, drop_z]
            self.data.qpos[qadr + 3 : qadr + 7] = quat
            zero_velocity(bid)
            mujoco.mj_forward(self.model, self.data)
            if settle_per_drop > 0:
                self._step_settle(settle_per_drop)
                # Zero velocities of all bodies after each step so prior drops
                # don't accumulate energy when later bodies land near them.
                for prior_bid in body_ids:
                    zero_velocity(prior_bid)
            placed_xys.append((x, y, r))
            samples.append(
                {
                    "body_name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid),
                    "drop_xy": [x, y],
                    "drop_z": drop_z,
                    "radius": r,
                    "yaw_deg": yaw_deg,
                    "roll_deg": roll_deg,
                    "pitch_deg": pitch_deg,
                    "final_pos": self.data.qpos[qadr : qadr + 3].tolist(),
                }
            )

        self._log_sample(
            action,
            {
                "region": region,
                "settle_per_drop": settle_per_drop,
                "edge_margin": edge_margin,
                "drops": samples,
            },
        )

    def _apply_grid_place(self, action: dict, snapshot: dict) -> None:
        """Battleship-style grid placement.

        See :mod:`mujoco_vr_teleop.placement_helpers` for the algorithm. The op
        config maps directly to :class:`placement_helpers.GridPlaceSpec`.
        """
        from mujoco_vr_teleop.placement_helpers import (
            GridPlaceSpec,
            place_on_grid,
        )
        from mujoco_vr_teleop.jenga_domain_randomization import surface_top

        body_ids = [
            bid
            for bid in self._resolve_bodies(action.get("target"))
            if self._body_freejoint_qadr(bid) is not None
        ]
        if not body_ids:
            return

        surface_geom_name, surface_z = surface_top(self, action, snapshot)

        region = action.get("region", {})
        region_x = tuple(region.get("x", (0.4, 0.9)))
        region_y = tuple(region.get("y", (-0.1, 0.5)))

        cell_size = action.get("cell_size", "auto")
        cell_size_value: float | None = None if cell_size == "auto" else float(cell_size)

        radii_entries = []
        for entry in action.get("radii", []) or []:
            pattern = entry.get("bodies_regex")
            radius = entry.get("radius")
            if pattern is not None and radius is not None:
                radii_entries.append((str(pattern), float(radius)))

        yaw_spec = action.get("yaw_deg", {"uniform": [-180.0, 180.0], "step": 10.0})
        roll_spec = action.get("roll_deg", 0.0)
        pitch_spec = action.get("pitch_deg", 0.0)

        def _range(spec):
            if isinstance(spec, dict) and "uniform" in spec:
                lo, hi = spec["uniform"]
                return float(lo), float(hi), float(spec.get("step", 0.0) or 0.0)
            return float(spec), float(spec), 0.0

        yaw_lo, yaw_hi, yaw_step = _range(yaw_spec)
        roll_lo, roll_hi, _ = _range(roll_spec)
        pitch_lo, pitch_hi, _ = _range(pitch_spec)

        spec = GridPlaceSpec(
            region_x=region_x,
            region_y=region_y,
            surface_z=surface_z,
            radii=radii_entries,
            default_radius=action.get("default_radius"),
            cell_size=cell_size_value,
            cell_gap=float(action.get("cell_gap", 0.01)),
            edge_margin=float(action.get("edge_margin", 0.02)),
            avoid_rects=action.get("avoid", []) or [],
            clearance=float(action.get("clearance", action.get("surface", {}).get("clearance", 0.003))),
            jitter_xy=float(action.get("jitter_xy", 0.0)),
            yaw_range_deg=(yaw_lo, yaw_hi),
            yaw_step_deg=yaw_step if yaw_step > 0 else 5.0,
            roll_range_deg=(roll_lo, roll_hi),
            pitch_range_deg=(pitch_lo, pitch_hi),
            settle_seconds=float(action.get("settle_seconds", 0.2)),
            settle_per_body=float(action.get("settle_per_body", 0.0)),
            min_cell_separation=int(action.get("min_cell_separation", 0)),
            hide_offset=tuple(action.get("hide_offset", (0.0, 0.0, -5.0))),
        )

        # Baseline quats from snapshot — these are the body's rest orientations
        # captured before any randomization. Composing sampled rotations onto
        # these preserves shapes like "bottle laying on its side".
        baseline_quats = {
            bid: np.asarray(snapshot["qpos"][self._body_freejoint_qadr(bid) + 3 : self._body_freejoint_qadr(bid) + 7], dtype=float).copy()
            for bid in body_ids
        }

        result = place_on_grid(
            self.model,
            self.data,
            self.rng,
            body_ids,
            spec,
            baseline_quats=baseline_quats,
        )
        result["surface_geom"] = surface_geom_name
        self._log_sample(action, result)
