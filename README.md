# 🚦 Otonom Araçlar İçin Gerçek Zamanlı Trafik Levhası Tespiti

<div align="center">
  <video src="assets/demo_video.mov" controls="controls" muted="muted" width="100%"></video>
</div>

Bu proje, otonom sürüş sistemleri için geliştirilmiş, gerçek zamanlı (real-time) trafik işareti tespit modelidir. YOLOv8 derin öğrenme mimarisi kullanılarak eğitilmiş olup, değişen ışık koşullarında ve hareket halindeyken yüksek tespit doğruluğu ve yüksek kare hızı (FPS) sunmaktadır.

## 📌 Proje Hakkında ve Mühendislik Yaklaşımı
Trafik ortamındaki zorlu koşullar ve donanımsal kısıtlamalar göz önüne alınarak, hız ve doğruluğun optimum dengesini sunan **YOLOv8s (Small)** modeli tercih edilmiştir. Proje geliştirme sürecinde modelin "ezberlemesini" (overfitting) önlemek ve doğruluğu artırmak için veri seti üzerinde iteratif optimizasyonlar (v5) yapılmıştır.

* **Veri Seti (v5):** Açık kaynaklı veri seti projeye özel optimize edilmiştir. Modelin kafasını karıştıran yabancı/gereksiz etiketler veri setinden kazınarak **30 kritik sınıfa** odaklanması sağlanmıştır. Ayrıca veri setine kendi görsellerimiz etiketlenerek eklenmiştir (49 adet). Modele özel hazırlanan bu veri setinin son sürümüne [Roboflow üzerinden buradan ulaşabilirsiniz](https://app.roboflow.com/ulas-tthvj/traffic-signs-and-traffic-lights-kt7qh/5).
* **Eğitim Süreci:** 50 Epoch, Google Colab (Tesla T4 GPU) ortamında eğitilmiştir. Eğitim süreci detayları için `notebooks/YOLOv8_trafik_egitim.ipynb` dosyasına göz atabilirsiniz.

## 📊 Model Başarısı (v5 Sonuçları)
Uygulanan veri seti optimizasyonları sonucunda (v5), model kapasitesinin zirvesine ulaşmış ve plato (plateau) evresine başarıyla girmiştir. Eğitim sonu elde edilen doğruluk metrikleri şöyledir:
* **mAP50 (Ortalama Hassasiyet):** > %90.4
* **Precision (Hassasiyet):** %91.5
* **Recall (Duyarlılık):** %85.1

*(Modelin hata kaybı (val_loss) grafikleri sıfır ezber (no overfitting) ile istikrarlı bir şekilde minimuma inmiştir. Epoch bazlı detaylı EKG/Analiz grafikleri için `metrics/results.png` ve `metrics/confusion_matrix.png` dosyalarını inceleyebilirsiniz.)*

## ⚙️ Kurulum (Local)

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

*(Kod yapısı; sistem donanımınızı dinamik olarak analiz ederek NVIDIA CUDA, Apple Silicon (MPS) veya standart CPU modunda en yüksek performansta çalışacak şekilde dizayn edilmiştir. Manuel bir donanım ayarı yapmanıza gerek yoktur.)*

## 🚀 Kullanım Senaryoları

Geliştirilen akıllı tespit algoritması (`detect.py`), verdiğiniz medya türünü (fotoğraf, video veya canlı yayın) otomatik olarak algılar, güncel `best_v5.pt` ağırlığıyla işler ve sonuçları `test_results/` klasörüne kaydeder. Test medyalarınızı proje dizinindeki `tests/` klasörüne ekleyebilirsiniz.

### Senaryo 1: Video Üzerinden Tespit
```bash
python detect.py --source tests/test1.mp4
```
**Sonuç:** Model videoyu anlık olarak ekranda işler ve bittiğinde `test_results/test1_processed.mp4` adıyla otomatik olarak kaydeder.

### Senaryo 2: Fotoğraf Üzerinden Tespit
```bash
python detect.py --source tests/test_foto.jpg
```
**Sonuç:** Fotoğraf tek seferde işlenir, tespitler çizilmiş haliyle ekranda gösterilir ve `test_results/test_foto_processed.jpg` adıyla kaydedilir.

### Senaryo 3: Canlı Web Kamerası ile Tespit
Gerçek zamanlı olarak bilgisayarınızın web kamerasını kullanmak için kaynak olarak `0` parametresini verin:
```bash
python detect.py --source 0
```
**Sonuç:** Kamera açılır, ekranda canlı tespit başlar. Kapatmak için klavyeden `q` tuşuna basmanız yeterlidir. Çıktı, `test_results/webcam_live_result.mp4` adıyla kaydedilir.

---

### 🎯 Gelişmiş Ayar: Olasılık Filtresi (`--conf`)
Modelin sadece belirli bir güven eşiğinin üzerindeki tespitleri ekrana çizmesini isterseniz `--conf` parametresini kullanabilirsiniz. Varsayılan değer **%25 (0.25)** olarak ayarlıdır.

**Örnek 1:** Kamerada sadece **%70** ve üzeri emin olduğu levhaları göstersin:
```bash
python detect.py --source 0 --conf 0.70
```

**Örnek 2:** Videoda sadece **%50** ve üzeri doğruluğa sahip levhaları işlesin:
```bash
python detect.py --source tests/test1.mp4 --conf 0.50
```