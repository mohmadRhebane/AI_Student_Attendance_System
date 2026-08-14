import os
import sys
import time
import ctypes
import traceback
from pathlib import Path


# يجب أن تبقى هذه القائمة موجودة طوال تشغيل البرنامج،
# وإلا قد يقوم Windows بإزالة مسارات DLL.
DLL_DIRECTORY_HANDLES = []


def configure_nvidia_dlls() -> None:
    print("=" * 70)
    print("STEP 1: Locating NVIDIA DLL directories")
    print("=" * 70)

    nvidia_root = (
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "nvidia"
    )

    print(f"Python environment: {sys.prefix}")
    print(f"NVIDIA root: {nvidia_root}")

    required_directories = [
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cufft" / "bin",
        nvidia_root / "nvjitlink" / "bin",
    ]

    optional_directories = [
        nvidia_root / "cuda_nvrtc" / "bin",
    ]

    missing_required = []

    for directory in required_directories:
        if directory.exists():
            print(f"FOUND   : {directory}")
        else:
            print(f"MISSING : {directory}")
            missing_required.append(directory)

    for directory in optional_directories:
        if directory.exists():
            print(f"FOUND   : {directory}")
        else:
            print(f"OPTIONAL: {directory}")

    if missing_required:
        missing_text = "\n".join(
            str(directory)
            for directory in missing_required
        )

        raise FileNotFoundError(
            "Required NVIDIA directories are missing:\n"
            f"{missing_text}"
        )

    existing_directories = [
        directory
        for directory in (
            required_directories
            + optional_directories
        )
        if directory.exists()
    ]

    # إضافة المجلدات إلى PATH قبل استيراد ONNX Runtime.
    old_path = os.environ.get("PATH", "")

    new_path = os.pathsep.join(
        str(directory)
        for directory in existing_directories
    )

    os.environ["PATH"] = (
        new_path
        + os.pathsep
        + old_path
    )

    # الاحتفاظ بالـhandles طوال تشغيل البرنامج.
    for directory in existing_directories:
        handle = os.add_dll_directory(str(directory))
        DLL_DIRECTORY_HANDLES.append(handle)


def test_required_dlls() -> None:
    print("\n" + "=" * 70)
    print("STEP 2: Loading required CUDA DLL files")
    print("=" * 70)

    required_dlls = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudnn64_9.dll",
        "cufft64_11.dll",
        "nvJitLink_120_0.dll",
    ]

    for dll_name in required_dlls:
        print(f"Loading {dll_name} ...", end=" ")

        ctypes.WinDLL(dll_name)

        print("OK")


def create_cuda_session():
    print("\n" + "=" * 70)
    print("STEP 3: Importing ONNX Runtime")
    print("=" * 70)

    # الاستيراد يجب أن يأتي بعد إعداد مسارات DLL.
    import onnxruntime as ort

    print(f"ONNX Runtime version : {ort.__version__}")
    print(f"ORT reported device  : {ort.get_device()}")
    print(
        "Available providers  : "
        f"{ort.get_available_providers()}"
    )

    if (
        "CUDAExecutionProvider"
        not in ort.get_available_providers()
    ):
        raise RuntimeError(
            "CUDAExecutionProvider is not available."
        )

    project_root = Path(__file__).resolve().parents[1]

    detector_path = (
        project_root
        / "models"
        / "scrfd_2.5g.onnx"
    )

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Detector not found: {detector_path}"
        )

    print("\n" + "=" * 70)
    print("STEP 4: Creating a real CUDA model session")
    print("=" * 70)

    print(f"Model: {detector_path}")

    session_options = ort.SessionOptions()

    # نعرض رسائل ONNX Runtime المهمة.
    session_options.log_severity_level = 2

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(detector_path),
        sess_options=session_options,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": "1",
                },
            ),
            "CPUExecutionProvider",
        ],
    )

    session_providers = session.get_providers()

    print(f"Session providers: {session_providers}")

    if not session_providers:
        raise RuntimeError(
            "No execution providers were activated."
        )

    if session_providers[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "The session fell back to CPU. "
            f"Active providers: {session_providers}"
        )

    return ort, session


def run_real_gpu_inference(ort, session) -> None:
    print("\n" + "=" * 70)
    print("STEP 5: Running real SCRFD inference")
    print("=" * 70)

    import numpy as np

    input_node = session.get_inputs()[0]

    print(f"Input name : {input_node.name}")
    print(f"Input shape: {input_node.shape}")
    print(f"Input type : {input_node.type}")

    dummy_frame = np.zeros(
        (1, 3, 640, 640),
        dtype=np.float32,
    )

    # Warm-up
    print("GPU warm-up ...")

    for _ in range(3):
        session.run(
            None,
            {input_node.name: dummy_frame},
        )

    repetitions = 10

    start = time.perf_counter()

    outputs = None

    for _ in range(repetitions):
        outputs = session.run(
            None,
            {input_node.name: dummy_frame},
        )

    elapsed_ms = (
        (time.perf_counter() - start)
        * 1000
        / repetitions
    )

    if outputs is None:
        raise RuntimeError(
            "The model returned no outputs."
        )

    print(f"Number of outputs: {len(outputs)}")

    for index, output in enumerate(outputs):
        print(
            f"Output {index}: shape={output.shape}"
        )

    print(
        f"Average inference latency: "
        f"{elapsed_ms:.3f} ms"
    )

    print("\n" + "=" * 70)
    print("GPU SMOKE TEST PASSED")
    print("=" * 70)

    print("SCRFD is running with CUDAExecutionProvider.")


def main() -> None:
    try:
        configure_nvidia_dlls()
        test_required_dlls()

        ort, session = create_cuda_session()

        run_real_gpu_inference(ort, session)

    except Exception as error:
        print("\n" + "=" * 70)
        print("GPU SMOKE TEST FAILED")
        print("=" * 70)

        print(f"Error type: {type(error).__name__}")
        print(f"Error: {error}")

        print("\nFull traceback:")
        traceback.print_exc()

        raise SystemExit(1)


if __name__ == "__main__":
    main()