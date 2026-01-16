import torch
import sys
import os

print("="*30)
print("SİSTEM ÖZETİ")
print("="*30)
print(f"Python Versiyonu : {sys.version.split()[0]}")
print(f"PyTorch Versiyonu: {torch.__version__}")
try:
    import tensorrt
    print(f"TensorRT Versiyonu: {tensorrt.__version__}")
except ImportError:
    print("TensorRT Versiyonu: KURULU DEĞİL veya BULUNAMADI")

print(f"CUDA Mevcut mu?  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Modeli       : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Versiyonu   : {torch.version.cuda}")
    # CuDNN versiyonu bazen kritik olabilir
    print(f"CuDNN Versiyonu  : {torch.backends.cudnn.version()}")

print("="*30)
print("KRİTİK KÜTÜPHANELER")
print("="*30)
# Kurulu paketleri pip ile listeler (özet)
os.system("pip list | findstr /i \"ultralytics torch tensorrt onnx\"")