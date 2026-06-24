#!/usr/bin/env python3
"""Probe which EGL device indices can actually build an offscreen framebuffer.

The RL rollout fails with `0x8cdd` (GL_FRAMEBUFFER_UNSUPPORTED) only when we pin
render to a NON-default EGL device. Single-process / default-device render works.
This script tells us, per EGL device index:
  - the CUDA GPU it maps to (EGL_CUDA_DEVICE_NV, if the driver exposes it)
  - whether a real offscreen FBO can be created on it (the exact thing robosuite
    does that throws 0x8cdd)

Run on the server:
    conda activate wilro
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python probe_egl.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import ctypes
from OpenGL import EGL
# EGL extension functions are NOT on OpenGL.EGL directly — import from submodules
# (same as robosuite's egl_context.py).
from OpenGL.EGL.EXT.device_base import EGLDeviceEXT
from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
try:
    from OpenGL.EGL.EXT.device_query import eglQueryDeviceAttribEXT
except Exception:  # some PyOpenGL builds put it under device_base
    eglQueryDeviceAttribEXT = getattr(EGL, "eglQueryDeviceAttribEXT", None)

EGL_CUDA_DEVICE_NV = 0x323A  # from EGL_NV_device_cuda


def list_egl_devices():
    max_dev = 16
    devices = (EGLDeviceEXT * max_dev)()
    num = EGL.EGLint()
    if not eglQueryDevicesEXT(max_dev, devices, ctypes.byref(num)):
        raise RuntimeError("eglQueryDevicesEXT failed")
    return [devices[i] for i in range(num.value)]


def cuda_index_of(dev):
    if eglQueryDeviceAttribEXT is None:
        return "n/a (no query)"
    try:
        val = EGL.EGLAttrib()
        if eglQueryDeviceAttribEXT(dev, EGL_CUDA_DEVICE_NV, ctypes.byref(val)):
            return int(val.value)
    except Exception as e:
        return f"n/a ({type(e).__name__})"
    return "n/a"


def try_offscreen_fbo(device_id, w=256, h=256):
    """Replicate robosuite's EGLGLContext + MjrContext FBO creation on device_id."""
    try:
        from robosuite.renderers.context.egl_context import EGLGLContext
        import mujoco
        ctx = EGLGLContext(max_width=w, max_height=h, device_id=device_id)
        ctx.make_current()
        # Minimal model so MjrContext (the offscreen FBO) actually builds — this
        # is the call that raises 0x8cdd.
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
        _ = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        return "OK"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"


def try_real_env(device_id):
    """Build a REAL LIBERO OffScreenRenderEnv on device_id (full scene + cameras),
    the exact thing the RL worker does — to tell whether non-zero GPUs can render
    the real env or only the probe's trivial model."""
    try:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero import get_libero_path, benchmark
        import os as _os
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = suite.get_task(0)
        bddl = _os.path.join(get_libero_path("bddl_files"),
                             task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                 camera_widths=256, control_freq=10,
                                 render_gpu_device_id=device_id)
        env.reset()
        env.close()
        return "OK (real env)"
    except Exception as e:
        return f"FAIL: {type(e).__name__}: {e}"


def main():
    import subprocess
    import sys
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    print("CUDA_DEVICE_ORDER    =", os.environ.get("CUDA_DEVICE_ORDER", "(unset)"))
    print("MUJOCO_EGL_DEVICE_ID =", os.environ.get("MUJOCO_EGL_DEVICE_ID", "(unset)"))
    print()
    devs = list_egl_devices()
    n = len(devs)
    # 1) CUDA mapping — printed from THIS process so it always shows.
    print(f"eglQueryDevicesEXT found {n} EGL device(s); egl_idx -> cuda_gpu:")
    for i, d in enumerate(devs):
        print(f"    egl_idx {i:>2}  ->  cuda_gpu {cuda_index_of(d)}")
    print()
    # 2) FBO test — each index in a fully isolated subprocess so a hard C-level
    #    abort (not a Python exception) is reported as a crash, not swallowed.
    print(f"{'egl_idx':>7}  result")
    print("-" * 60)
    for i in range(n):
        p = subprocess.run([sys.executable, __file__, "--fbo", str(i)],
                           capture_output=True, text=True,
                           env={**os.environ, "MUJOCO_GL": "egl",
                                "PYOPENGL_PLATFORM": "egl"})
        out = (p.stdout + p.stderr).strip().replace("\n", " | ")
        if p.returncode != 0 and "RESULT:" not in p.stdout:
            res = f"CRASH (rc={p.returncode}) {out[-200:]}"
        else:
            res = p.stdout.split("RESULT:", 1)[-1].strip() if "RESULT:" in p.stdout else out[-200:]
        print(f"{i:>7}  {res}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--fbo":
        idx = int(sys.argv[2])
        print("RESULT:" + try_offscreen_fbo(idx), flush=True)
    elif len(sys.argv) == 3 and sys.argv[1] == "--realenv":
        idx = int(sys.argv[2])
        print(f"real LiberoEnv on device {idx}: " + try_real_env(idx), flush=True)
    else:
        main()
