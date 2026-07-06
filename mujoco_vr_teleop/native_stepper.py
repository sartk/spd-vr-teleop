from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


def _compile_native_stepper() -> Path:
    src = Path(__file__).with_name("native_stepper.c")
    if not src.exists():
        raise FileNotFoundError(f"Missing native stepper source: {src}")

    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise RuntimeError("No C compiler found (`cc`, `gcc`, or `clang`)")

    mujoco_root = Path(mujoco.__file__).resolve().parent
    include_dir = mujoco_root / "include"
    lib_candidates = sorted(mujoco_root.glob("libmujoco.so*")) + sorted(
        mujoco_root.glob("libmujoco*.dylib")
    )
    if not lib_candidates:
        raise RuntimeError(f"Could not find libmujoco in {mujoco_root}")
    lib_path = lib_candidates[0]

    build_dir = Path(__file__).resolve().parent / ".native"
    build_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    digest.update(src.read_bytes())
    digest.update(str(lib_path).encode("utf-8"))
    digest.update(mujoco.__version__.encode("utf-8"))
    out_path = build_dir / f"native_stepper_{digest.hexdigest()[:12]}.so"
    if out_path.exists():
        return out_path

    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c11",
        f"-I{include_dir}",
        str(src),
        str(lib_path),
        f"-Wl,-rpath,{mujoco_root}",
        "-lm",
        "-o",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build native MuJoCo stepper:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # On macOS, the wheel's libmujoco records its install_name as a path inside
    # a mujoco.framework that the wheel does not actually ship. Rewrite the
    # recorded dependency to point at the dylib's real location so dyld can
    # resolve it at load time.
    if sys.platform == "darwin":
        install_name = subprocess.run(
            ["otool", "-D", str(lib_path)], capture_output=True, text=True, check=True
        ).stdout.splitlines()[-1].strip()
        subprocess.run(
            ["install_name_tool", "-change", install_name, str(lib_path), str(out_path)],
            check=True,
        )

    return out_path


class NativeStepper:
    def __init__(self):
        lib_path = _compile_native_stepper()
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.mjvr_step_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._lib.mjvr_step_model.restype = None
        self._lib.mjvr_pack_body_transforms.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.mjvr_pack_body_transforms.restype = None

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, n_substeps: int) -> None:
        self._lib.mjvr_step_model(
            ctypes.c_void_p(model._address),
            ctypes.c_void_p(data._address),
            int(n_substeps),
        )

    def pack_body_transforms(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_ids: np.ndarray,
        out: np.ndarray,
    ) -> None:
        body_ids = np.ascontiguousarray(body_ids, dtype=np.int32)
        out = np.ascontiguousarray(out, dtype=np.float32)
        expected_size = int(body_ids.size) * 8
        if out.size != expected_size:
            raise ValueError(f"Expected output buffer of size {expected_size}, got {out.size}")

        self._lib.mjvr_pack_body_transforms(
            ctypes.c_void_p(model._address),
            ctypes.c_void_p(data._address),
            body_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(body_ids.size),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )


__all__ = ["NativeStepper"]
