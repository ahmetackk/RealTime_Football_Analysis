from ultralytics import YOLO
import torch
import time
import numpy as np

# Modeller
PT_MODEL = "models/player_detection.pt"
ENGINE_MODEL = "models/player_detection.engine"

def benchmark_pure(model_path, name):
    print(f"\n--- {name} Saf Hız Testi ---")
    try:
        model = YOLO(model_path, task='detect')
    except:
        print(f"Model yüklenemedi: {model_path}")
        return

    # Rastgele bir resim tensörü oluştur (Video okuma yok, sadece işlem)
    # 1 adet, 3 kanallı, 640x640 boyutunda sahte resim
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Isınma turları
    print("GPU Isınıyor...")
    for _ in range(20):
        model.predict(dummy_input, verbose=False, device=0, half=True)

    print("Test Başlıyor (500 Tekrar)...")
    start = time.time()
    for _ in range(500):
        # Kaydetme ve yazdırma kapalı
        model.predict(dummy_input, verbose=False, device=0, half=True)
    end = time.time()

    duration = end - start
    fps = 500 / duration
    print(f"Bitiş Süresi: {duration:.4f} sn")
    print(f">> {name} HIZI: {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        fps_pt = benchmark_pure(PT_MODEL, "PyTorch")
        fps_engine = benchmark_pure(ENGINE_MODEL, "TensorRT")
        
        if fps_engine and fps_pt:
            print(f"\nSAF GPU FARKI: {fps_engine/fps_pt:.2f}x daha hızlı")
    else:
        print("Hata: GPU bulunamadı.")