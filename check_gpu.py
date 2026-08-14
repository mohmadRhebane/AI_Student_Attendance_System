import os
import site

# 1. إخبار ويندوز بمكان ملفات CUDA التي حملناها عبر pip
try:
    site_packages = site.getsitepackages()[0]
    cuda_paths = [
        os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)
except Exception as e:
    print(f"Warning during DLL linking: {e}")

# 2. استدعاء مكتبة الذكاء الاصطناعي بعد ربط المسارات
import onnxruntime as ort

print(f"ORT Version: {ort.__version__}")
providers = ort.get_available_providers()
print(f"Available Providers: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("\n✅ SUCCESS: CUDA is ready! (RTX 3050 is connected)")
else:
    print("\n❌ FAILED: CPU is being used instead of GPU.")