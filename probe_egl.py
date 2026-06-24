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


def main():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    print("CUDA_DEVICE_ORDER    =", os.environ.get("CUDA_DEVICE_ORDER", "(unset)"))
    print("MUJOCO_EGL_DEVICE_ID =", os.environ.get("MUJOCO_EGL_DEVICE_ID", "(unset)"))
    print()
    devs = list_egl_devices()
    print(f"eglQueryDevicesEXT found {len(devs)} EGL device(s)\n")
    print(f"{'egl_idx':>7}  {'cuda_gpu':>8}  result")
    print("-" * 50)
    for i, d in enumerate(devs):
        cuda = cuda_index_of(d)
        # Each FBO attempt in a forked child so a FatalError can't poison the rest.
        pid = os.fork()
        if pid == 0:
            res = try_offscreen_fbo(i)
            print(f"{i:>7}  {str(cuda):>8}  {res}", flush=True)
            os._exit(0)
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main()
