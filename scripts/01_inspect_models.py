import os
import sys
from pathlib import Path

# هذه القائمة ضرورية لكي لا يقوم ويندوز بحذف المسارات من الذاكرة
DLL_DIRECTORY_HANDLES = []

def configure_cuda_dll_paths() -> None:
    # تحديد مسار ملفات NVIDIA داخل بيئتك الافتراضية
    nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    
    dll_directories = [
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
    ]

    existing_directories = [d for d in dll_directories if d.exists()]
    
    print("\n" + "="*50)
    print("🔍 جاري البحث عن ملفات CUDA (DLLs):")
    for directory in existing_directories:
        print(f"✅ FOUND: {directory.name}")
        
    if not existing_directories:
        print(f"❌ لم أجد ملفات NVIDIA في المسار: {nvidia_root}")
        return

    # إضافتها إلى مسارات النظام الوهمية
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(str(d) for d in existing_directories) + os.pathsep + current_path

    # أمر مهم جداً لويندوز للاحتفاظ بالمسار
    for directory in existing_directories:
        handle = os.add_dll_directory(str(directory))
        DLL_DIRECTORY_HANDLES.append(handle)

# ==========================================
# ⚠️ يجب أن نستدعي هذه الدالة قبل استدعاء ONNX
# ==========================================
configure_cuda_dll_paths()

import onnxruntime as ort

def inspect_model(model_path):
    print(f"\n{'=' * 50}")
    print(f"🔬 فحص النموذج: {model_path.name}")
    print(f"{'=' * 50}")

    if not model_path.exists():
        print(f"❌ الملف غير موجود: {model_path}")
        return

    available_providers = ort.get_available_providers()
    
    if "CUDAExecutionProvider" not in available_providers:
        print("❌ لم يتم العثور على CUDAExecutionProvider. النظام سيستخدم الـ CPU.")
        return

    # إعدادات خاصة لتحسين أداء كرت الشاشة
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[
                ("CUDAExecutionProvider", {"device_id": "0"}),
                "CPUExecutionProvider",
            ],
        )
    except Exception as e:
        print(f"❌ فشل تشغيل النموذج على كرت الشاشة: {e}")
        return

    active_providers = session.get_providers()
    print(f"✅ تم تحميل النموذج بنجاح باستخدام: {active_providers[0]}\n")

    print("📥 المدخلات (Inputs):")
    for input_node in session.get_inputs():
        print(f"  - {input_node.name} | Shape: {input_node.shape} | Type: {input_node.type}")

    print("\n📤 المخرجات (Outputs):")
    for output_node in session.get_outputs():
        print(f"  - {output_node.name} | Shape: {output_node.shape}")

def main():
    models_dir = Path("models")
    detector_path = models_dir / "scrfd_2.5g.onnx"
    recognizer_path = models_dir / "resnet50_arcface.onnx"

    inspect_model(detector_path)
    inspect_model(recognizer_path)

if __name__ == "__main__":
    main()