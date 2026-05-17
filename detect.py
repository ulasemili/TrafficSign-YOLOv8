import cv2
import torch
from ultralytics import YOLO

# 1. Dinamik Donanım Kontrolü (Universal Device Selection)
if torch.cuda.is_available():
    device = "cuda"
    print("[BİLGİ] NVIDIA GPU (CUDA) tespit edildi. Süreç hızlandırılıyor...")
elif torch.backends.mps.is_available():
    device = "mps"
    print("[BİLGİ] Apple Silicon (MPS) tespit edildi. Süreç hızlandırılıyor...")
else:
    device = "cpu"
    print("[BİLGİ] Harici GPU bulunamadı. Standart İşlemci (CPU) kullanılacak.")

# 2. Modeli Yükle
print("[BİLGİ] YOLOv8 Modeli yükleniyor...")
model = YOLO("best.pt")

# 3. İşlenecek videonun yolunu belirt (Test videonun adını buraya yaz)
video_path = "test_video.mp4" 
cap = cv2.VideoCapture(video_path)

print("[BİLGİ] Video işleme başlıyor. Çıkmak için klavyeden 'q' tuşuna basın.")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 4. Modeli kare üzerinde dinamik donanımla çalıştır
        results = model(frame, device=device) 

        # 5. Sonuçları çizilmiş kareyi al
        annotated_frame = results[0].plot()

        # 6. Ekranda Göster
        cv2.imshow("Oruntu Tanima - Gercek Zamanli Trafik Levhasi Tespiti", annotated_frame)

        # Çıkmak için 'q' tuşuna bas
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
print("[BİLGİ] İşlem başarıyla tamamlandı.")