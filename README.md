# Otonom Araçlar İçin Gerçek Zamanlı Trafik Levhası Tespiti 🚦

Bu proje, otonom sürüş sistemleri için geliştirilmiş, gerçek zamanlı (real-time) trafik işareti tespit modelidir. YOLOv8 derin öğrenme mimarisi kullanılarak eğitilmiş olup, yüksek tespit doğruluğu (mAP50) ve yüksek kare hızı (FPS) sunmaktadır.

## 📌 Proje Hakkında
Trafik ortamındaki değişken ışık koşulları, hareket bulanıklığı ve donanımsal kısıtlamalar göz önüne alınarak, hız ve doğruluğun optimum dengesini sunan **YOLOv8s (Small)** modeli tercih edilmiştir. 
* **Veri Seti:** +6900 görsel (Roboflow üzerinden ön işleme yapılmıştır).
* **Sınıf Sayısı:** 22 Kritik Trafik Levhası (Yaya Geçidi, Kırmızı Işık, Hız Sınırları vb.)
* **Eğitim Süreci:** 50 Epoch, Google Colab (Tesla T4 GPU)

## 📊 Model Başarısı 
Model eğitim süreci sonunda elde edilen doğruluk metrikleri şöyledir:
* **mAP50 (Ortalama Hassasiyet):** `%87.5`
* **Precision (Hassasiyet):** `%82.9`
* **Recall (Duyarlılık):** `%83.4`

*(Modelin özellikle "Yaya Geçidi" (%99.1) ve "Kırmızı Işık" (%92.4) tespitlerindeki doğruluğu otonom sürüş güvenliği standartlarındadır.)*

## ⚙️ Kurulum ve Çalıştırma (Lokal PC)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli kütüphaneleri kurun: pip install -r requirements.txt
1. Modeli test videonuz üzerinde çalıştırın: python detect.py

*(Not: Sisteminizdeki donanıma göre kod otomatik olarak NVIDIA CUDA, Apple MPS veya standart CPU modunda çalışacak şekilde optimize edilmiştir. Manuel bir ayar yapmanıza gerek yoktur.)*