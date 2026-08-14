# import os
# import sys
# from pathlib import Path


# _DLL_DIRECTORY_HANDLES = []
# _DLLS_CONFIGURED = False


# def configure_cuda_dlls() -> None:
#     """
#     Register NVIDIA DLL directories before importing ONNX Runtime.

#     On Windows with Python-installed NVIDIA packages, CUDA DLL folders
#     are not always included automatically in the DLL search path.
#     """

#     global _DLLS_CONFIGURED

#     if _DLLS_CONFIGURED:
#         return

#     # على Linux نترك النظام وONNX Runtime يعالجان المسارات.
#     if os.name != "nt":
#         _DLLS_CONFIGURED = True
#         return

#     nvidia_root = (
#         Path(sys.prefix)
#         / "Lib"
#         / "site-packages"
#         / "nvidia"
#     )

#     if not nvidia_root.exists():
#         raise RuntimeError(
#             f"NVIDIA package directory does not exist: {nvidia_root}"
#         )

#     # اكتشاف جميع مجلدات bin تلقائيًا:
#     # cuda_runtime, cublas, cudnn, cufft, nvjitlink...
#     dll_directories = sorted(
#         directory
#         for directory in nvidia_root.glob("*/bin")
#         if directory.is_dir()
#     )

#     if not dll_directories:
#         raise RuntimeError(
#             f"No NVIDIA bin directories found under: {nvidia_root}"
#         )

#     dll_path = os.pathsep.join(
#         str(directory)
#         for directory in dll_directories
#     )

#     current_path = os.environ.get("PATH", "")

#     os.environ["PATH"] = (
#         dll_path
#         + os.pathsep
#         + current_path
#     )

#     for directory in dll_directories:
#         handle = os.add_dll_directory(str(directory))
#         _DLL_DIRECTORY_HANDLES.append(handle)

#     _DLLS_CONFIGURED = True


# # يجب تنفيذه قبل import onnxruntime.
# configure_cuda_dlls()


# import onnxruntime as ort


# def get_runtime_info() -> dict:
#     return {
#         "onnxruntime_version": ort.__version__,
#         "device": ort.get_device(),
#         "available_providers": ort.get_available_providers(),
#     }


# def create_gpu_session(model_path):
#     """
#     Create an optimized ONNX Runtime session with CUDA as the primary
#     execution provider. CPU is retained only as a secondary provider.
#     """

#     resolved_model_path = Path(model_path).resolve()

#     if not resolved_model_path.exists():
#         raise FileNotFoundError(
#             f"ONNX model not found: {resolved_model_path}"
#         )

#     available_providers = ort.get_available_providers()

#     if "CUDAExecutionProvider" not in available_providers:
#         raise RuntimeError(
#             "CUDAExecutionProvider is unavailable. "
#             f"Available providers: {available_providers}"
#         )

#     session_options = ort.SessionOptions()

#     session_options.graph_optimization_level = (
#         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
#     )

#     # 2 يعرض التحذيرات والأخطاء ويخفي رسائل Info الطويلة.
#     session_options.log_severity_level = 2

#     session = ort.InferenceSession(
#         str(resolved_model_path),
#         sess_options=session_options,
#         providers=[
#             (
#                 "CUDAExecutionProvider",
#                 {
#                     "device_id": "0",
#                     "cudnn_conv_algo_search": "HEURISTIC",
#                     "do_copy_in_default_stream": "1",
#                 },
#             ),
#             "CPUExecutionProvider",
#         ],
#     )

#     active_providers = session.get_providers()

#     if not active_providers:
#         raise RuntimeError(
#             f"No execution provider loaded for: {resolved_model_path}"
#         )

#     if active_providers[0] != "CUDAExecutionProvider":
#         raise RuntimeError(
#             "CUDA initialization failed and the model fell back to CPU. "
#             f"Model: {resolved_model_path}; "
#             f"active providers: {active_providers}"
#         )

#     return session

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional


_DLL_DIRECTORY_HANDLES = []
_DLLS_CONFIGURED = False
_DLL_CONFIGURATION_ERROR: Optional[str] = None
_ORT = None


def configure_cuda_dlls(
    *,
    strict: bool = True,
) -> bool:
    """
    Configure NVIDIA DLL directories on Windows.

    Returns:
        True:
            CUDA DLL setup succeeded or is not required.

        False:
            CUDA DLL setup failed, but strict=False allows
            the caller to continue and possibly fall back to CPU.
    """

    global _DLLS_CONFIGURED
    global _DLL_CONFIGURATION_ERROR

    if _DLLS_CONFIGURED:
        return True

    # Linux normally resolves CUDA libraries through the system.
    if os.name != "nt":
        _DLLS_CONFIGURED = True
        return True

    nvidia_root = (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "nvidia"
    )

    if not nvidia_root.exists():
        message = (
            "NVIDIA package directory does not exist: "
            f"{nvidia_root}"
        )

        _DLL_CONFIGURATION_ERROR = message

        if strict:
            raise RuntimeError(message)

        return False

    dll_directories = sorted(
        directory
        for directory in nvidia_root.glob("*/bin")
        if directory.is_dir()
    )

    if not dll_directories:
        message = (
            "No NVIDIA bin directories found under: "
            f"{nvidia_root}"
        )

        _DLL_CONFIGURATION_ERROR = message

        if strict:
            raise RuntimeError(message)

        return False

    dll_path = os.pathsep.join(
        str(directory)
        for directory in dll_directories
    )

    current_path = os.environ.get(
        "PATH",
        "",
    )

    os.environ["PATH"] = (
        dll_path
        + os.pathsep
        + current_path
    )

    for directory in dll_directories:
        try:
            handle = os.add_dll_directory(
                str(directory)
            )
            _DLL_DIRECTORY_HANDLES.append(
                handle
            )
        except (AttributeError, OSError):
            # PATH was already updated above.
            pass

    _DLLS_CONFIGURED = True
    _DLL_CONFIGURATION_ERROR = None

    return True


def _load_onnxruntime(
    *,
    prefer_cuda: bool,
    allow_cpu_fallback: bool,
):
    """
    Import ONNX Runtime lazily.

    CUDA DLL setup is attempted before importing ONNX Runtime
    when CUDA is requested.
    """

    global _ORT

    if _ORT is not None:
        return _ORT

    if prefer_cuda:
        configure_cuda_dlls(
            strict=not allow_cpu_fallback,
        )

    import onnxruntime as ort

    _ORT = ort

    return ort


def get_runtime_info() -> dict:
    """
    Return information about the ONNX Runtime environment.

    This function does not require CUDA to be present.
    """

    ort = _load_onnxruntime(
        prefer_cuda=False,
        allow_cpu_fallback=True,
    )

    return {
        "onnxruntime_version": ort.__version__,
        "device": ort.get_device(),
        "available_providers": (
            ort.get_available_providers()
        ),
        "cuda_dlls_configured": (
            _DLLS_CONFIGURED
        ),
        "cuda_configuration_error": (
            _DLL_CONFIGURATION_ERROR
        ),
    }


def create_inference_session(
    model_path,
    *,
    provider: str = "cuda",
    allow_cpu_fallback: bool = False,
    device_id: int = 0,
):
    """
    Create an ONNX Runtime inference session.

    Supported provider modes:

        cuda:
            Prefer CUDA. CPU fallback depends on
            allow_cpu_fallback.

        cpu:
            Force CPUExecutionProvider.

        auto:
            Prefer CUDA when available, otherwise use CPU.

    Args:
        model_path:
            Path to the ONNX model.

        provider:
            "cuda", "cpu", or "auto".

        allow_cpu_fallback:
            When True, CPU may be used when CUDA is unavailable
            or CUDA session initialization fails.

        device_id:
            CUDA GPU index.

    Returns:
        onnxruntime.InferenceSession
    """

    resolved_model_path = Path(
        model_path
    ).resolve()

    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            "ONNX model not found: "
            f"{resolved_model_path}"
        )

    requested_provider = (
        str(provider)
        .strip()
        .lower()
    )

    if requested_provider not in {
        "cuda",
        "cpu",
        "auto",
    }:
        raise ValueError(
            "provider must be one of: "
            "'cuda', 'cpu', 'auto'. "
            f"Received: {provider!r}"
        )

    # AUTO always allows CPU because that is its purpose.
    effective_cpu_fallback = (
        bool(allow_cpu_fallback)
        or requested_provider == "auto"
    )

    prefer_cuda = requested_provider in {
        "cuda",
        "auto",
    }

    cuda_setup_ok = True

    if prefer_cuda:
        try:
            cuda_setup_ok = configure_cuda_dlls(
                strict=not effective_cpu_fallback,
            )
        except Exception:
            if not effective_cpu_fallback:
                raise

            cuda_setup_ok = False

    ort = _load_onnxruntime(
        prefer_cuda=False,
        allow_cpu_fallback=True,
    )

    available_providers = (
        ort.get_available_providers()
    )

    if (
        "CPUExecutionProvider"
        not in available_providers
    ):
        if requested_provider == "cpu":
            raise RuntimeError(
                "CPUExecutionProvider is unavailable. "
                "Available providers: "
                f"{available_providers}"
            )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Warnings + errors.
    session_options.log_severity_level = 2

    # ----------------------------------------------------------
    # CPU explicitly requested
    # ----------------------------------------------------------

    if requested_provider == "cpu":
        return ort.InferenceSession(
            str(resolved_model_path),
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

    # ----------------------------------------------------------
    # CUDA / AUTO
    # ----------------------------------------------------------

    cuda_available = (
        cuda_setup_ok
        and "CUDAExecutionProvider"
        in available_providers
    )

    if not cuda_available:
        if not effective_cpu_fallback:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable "
                "and CPU fallback is disabled. "
                f"Available providers: "
                f"{available_providers}"
            )

        return ort.InferenceSession(
            str(resolved_model_path),
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

    cuda_provider = (
        "CUDAExecutionProvider",
        {
            "device_id": str(
                int(device_id)
            ),
            "cudnn_conv_algo_search": (
                "HEURISTIC"
            ),
            "do_copy_in_default_stream": "1",
        },
    )

    requested_providers = [
        cuda_provider,
    ]

    if effective_cpu_fallback:
        requested_providers.append(
            "CPUExecutionProvider"
        )

    try:
        session = ort.InferenceSession(
            str(resolved_model_path),
            sess_options=session_options,
            providers=requested_providers,
        )

    except Exception as cuda_error:
        if not effective_cpu_fallback:
            raise RuntimeError(
                "CUDA session initialization failed "
                "and CPU fallback is disabled. "
                f"Model: {resolved_model_path}"
            ) from cuda_error

        session = ort.InferenceSession(
            str(resolved_model_path),
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider",
            ],
        )

    active_providers = (
        session.get_providers()
    )

    if not active_providers:
        raise RuntimeError(
            "No execution provider loaded for: "
            f"{resolved_model_path}"
        )

    # Strict CUDA means CUDA must actually be first.
    if (
        requested_provider == "cuda"
        and not effective_cpu_fallback
        and active_providers[0]
        != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "CUDA initialization failed and "
            "CPU fallback is disabled. "
            f"Model: {resolved_model_path}; "
            f"active providers: "
            f"{active_providers}"
        )

    return session


def create_gpu_session(
    model_path,
    *,
    provider: str = "cuda",
    allow_cpu_fallback: bool = False,
    device_id: int = 0,
):
    """
    Backwards-compatible wrapper.

    Older project modules use create_gpu_session().
    New code may use create_inference_session().
    """

    return create_inference_session(
        model_path,
        provider=provider,
        allow_cpu_fallback=(
            allow_cpu_fallback
        ),
        device_id=device_id,
    )