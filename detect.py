import cv2
import torch
import argparse
import os
from ultralytics import YOLO

# 1. Terminalden Gelen Komutları Yakalama
parser = argparse.ArgumentParser(description="YOLOv8 Traffic Sign Detection v5")
parser.add_argument("--source", type=str, default="0", help="Medya yolu veya Webcam için '0'")
parser.add_argument("--conf", type=float, default=0.25, help="Minimum tespit olasılık eşiği (Örn: 0.50)")
args = parser.parse_args()

# 2. Klasör Mimarisi
input_folder = "tests"
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

# 4. Modeli Yükle (GÜNCELLENDİ: v5 Modeli)
print("[BİLGİ] YOLOv8 Modeli (v5) yükleniyor...")
model = YOLO("best_v5.pt")

# 5. Dosya Formatını Analiz Etme
source = args.source
_, ext = os.path.splitext(source)
is_image = ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

if is_image:
    # ================= FOTOĞRAF İŞLEME BLOĞU =================
    print(f"[BİLGİ] Fotoğraf okuma başlatıldı: {source} (Eşik: {args.conf})")
    base_name = os.path.basename(source)
    file_name, file_ext = os.path.splitext(base_name)
    output_filename = f"{file_name}_processed{file_ext}"
    output_path = os.path.join(output_folder, output_filename)

    frame = cv2.imread(source)
    
    if frame is None:
        print(f"[HATA] Fotoğraf okunamadı: {source}")
    else:
        # Modeli belirlenmiş olasılık eşiği (conf) ile çalıştırıyoruz
        results = model(frame, device=device, conf=args.conf)
        annotated_frame = results[0].plot()

        cv2.imwrite(output_path, annotated_frame)
        print(f"✅ [BAŞARILI] Fotoğraf hazır: {output_path}")

        cv2.imshow("Traffic Sign Detection - Image Result", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    # ================= VİDEO VE WEBCAM İŞLEME BLOĞU =================
    if source.isdigit():
        source = int(source)
        print(f"[BİLGİ] Canlı Webcam akışı başlatılıyor... (Eşik: {args.conf})")
        output_filename = "webcam_live_result.mp4"
    else:
        print(f"[BİLGİ] Video okuma başlatıldı: {source} (Eşik: {args.conf})")
        base_name = os.path.basename(source)
        file_name, _ = os.path.splitext(base_name)
        output_filename = f"{file_name}_processed.mp4"

    output_path = os.path.join(output_folder, output_filename)
    cap = cv2.VideoCapture(source)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Modeli belirlenmiş olasılık eşiği (conf) ile çalıştırıyoruz
            results = model(frame, device=device, conf=args.conf) 
            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            cv2.imshow("Real-Time Traffic Sign Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    print(f"✅ [BAŞARILI] Video hazır: {output_path}")