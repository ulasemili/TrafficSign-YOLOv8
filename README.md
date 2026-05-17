# 🚦 Otonom Araçlar İçin Gerçek Zamanlı Trafik Levhası Tespiti

Bu proje, otonom sürüş sistemleri için geliştirilmiş, gerçek zamanlı (real-time) trafik işareti tespit modelidir. YOLOv8 derin öğrenme mimarisi kullanılarak eğitilmiş olup, değişen ışık koşullarında ve hareket halindeyken yüksek tespit doğruluğu ve yüksek kare hızı (FPS) sunmaktadır.

## 📌 Proje Hakkında
Trafik ortamındaki zorlu koşullar ve donanımsal kısıtlamalar göz önüne alınarak, hız ve doğruluğun optimum dengesini sunan **YOLOv8s (Small)** modeli tercih edilmiştir. 
* **Veri Seti:** Açık kaynaklı trafik veri seti üzerinden türetilmiş olup; projemiz için optimize edilerek gereksiz etiketlerden arındırılmış, 23 kritik sınıfa indirgenmiştir. Modele özel hazırladığımız bu veri setine [Roboflow üzerinden buradan ulaşabilirsiniz](https://app.roboflow.com/ulas-tthvj/traffic-signs-and-traffic-lights-kt7qh/2).
* **Sınıf Sayısı:** 23 Kritik Trafik Levhası (Yaya Geçidi, Kırmızı Işık, Hız Sınırları vb.)
* **Eğitim Süreci:** 50 Epoch, Google Colab (Tesla T4 GPU) ortamında eğitilmiştir. Eğitim süreci detayları için `notebooks/` klasörüne göz atabilirsiniz.

## 📊 Model Başarısı 
Model eğitim süreci sonunda elde edilen doğruluk metrikleri şöyledir:
* **mAP50 (Ortalama Hassasiyet):** %87.5
* **Precision (Hassasiyet):** %82.9
* **Recall (Duyarlılık):** %83.4

*(Modelin özellikle "Yaya Geçidi" (%99.1) ve "Kırmızı Işık" (%92.4) tespitlerindeki doğruluğu otonom sürüş güvenliği standartlarındadır. Detaylı analiz grafikleri için `metrics/` klasörünü inceleyebilirsiniz.)*

## ⚙️ Kurulum (Lokal PC)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Repoyu bilgisayarınıza klonlayın ve klasöre girin:
```bash
git clone https://github.com/ulasemili/TrafficSign-YOLOv8.git
cd TrafficSign-YOLOv8
```

2. Gerekli kütüphaneleri kurun:
```bash
pip install -r requirements.txt
```

*(Sistem donanımınıza göre kod otomatik olarak NVIDIA CUDA, Apple MPS veya standart CPU modunda çalışacak şekilde dizayn edilmiştir. Manuel bir ayar yapmanıza gerek yoktur.)*

## 🚀 Kullanım Senaryoları

Yazdığımız yapay zeka entegrasyonu, kaynak dosyalarını otomatik olarak işler ve sonuçları ayrı bir klasöre (`test_results`) kaydeder.

### Senaryo 1: Video Üzerinden Tespit (Önerilen)
Dışarıda çektiğiniz test videolarını proje dizininde otomatik oluşan `test_videos` klasörünün içine atın ve terminalden şu komutu çalıştırın:
```bash
python detect.py --source test_videos/test1.mp4
```
**Sonuç:** Model videoyu anlık olarak ekranda işler ve bittiğinde `test_results/` klasörüne `test1_processed.mp4` adıyla otomatik olarak kaydeder.

### Senaryo 2: Canlı Web Kamerası ile Tespit
Gerçek zamanlı olarak bilgisayarınızın web kamerasını (veya bağlı bir akıllı telefonu) kullanarak canlı tespit yapmak için kaynak olarak `0` parametresini verin:
```bash
python detect.py --source 0
```
**Sonuç:** Kamera açılır, ekranda canlı tespit başlar. Kapatmak için `q` tuşuna basmanız yeterlidir. Çıktı, otomatik olarak `test_results/webcam_live_result.mp4` adıyla kaydedilir.