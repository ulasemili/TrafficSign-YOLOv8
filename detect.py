import cv2
import torch
import argparse
import os
from ultralytics import YOLO

# 1. Terminalden Gelen Komutları Yakalama
parser = argparse.ArgumentParser(description="YOLOv8 Traffic Sign Detection")
parser.add_argument("--source", type=str, default="0", help="Videonun yolu veya Webcam için '0' yazın")
args = parser.parse_args()

# 2. Klasör Mimarisi (İngilizce Standartlar)
input_folder = "test_videos"
output_folder = "test_results"
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 3. Dinamik Donanım Kontrolü
if torch.cuda.is_available():
    device = "cuda"
    print("[BİLGİ] NVIDIA GPU (CUDA) tespit edildi.")
elif torch.backends.mps.is_available():
    device = "mps"
    print("[BİLGİ] Apple Silicon (MPS) tespit edildi. Süreç hızlandırılıyor...")
else:
    device = "cpu"
    print("[BİLGİ] Harici GPU bulunamadı. CPU kullanılacak.")

# 4. Modeli Yükle
print("[BİLGİ] YOLOv8 Modeli yükleniyor...")
model = YOLO("best.pt")

# 5. Kaynak ve Çıktı Dosya İsimlendirme Mantığı (İngilizce Uzantılar)
source = args.source
if source.isdigit():
    source = int(source)
    print("[BİLGİ] Canlı Webcam akışı başlatılıyor...")
    output_filename = "webcam_live_result.mp4"
else:
    print(f"[BİLGİ] Video okuma başlatıldı: {source}")
    base_name = os.path.basename(source)
    file_name, _ = os.path.splitext(base_name)
    output_filename = f"{file_name}_processed.mp4"

# Çıktının tam kaydedileceği yol
output_path = os.path.join(output_folder, output_filename)

# 6. Video Okuyucu ve Kaydedici Ayarları
cap = cv2.VideoCapture(source)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("[BİLGİ] İşlem başladı. Ekranda izleyebilirsiniz. Çıkmak veya erken bitirmek için 'q' tuşuna basın.")
print(f"[BİLGİ] Çıktı dosyanız şu konuma kaydedilecek: {output_path}")

# 7. İşleme Döngüsü
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, device=device) 
        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        cv2.imshow("Real-Time Traffic Sign Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 8. Temizlik ve Dosyayı Kaydetme
cap.release()
out.release() 
cv2.destroyAllWindows()
print(f"✅ [BAŞARILI] İşlem tamamlandı! Video hazır: {output_path}")