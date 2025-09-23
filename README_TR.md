# Sürücü Tanıma Otomasyonu  

Bu proje, **Python** ve derin öğrenme kütüphaneleri (**FaceNet, MTCNN**) kullanarak araçlardaki sürücüleri otomatik olarak tespit eden bir otomasyon sistemidir. Gelişmiş yüz tanıma algoritmaları sayesinde hem canlı kamera akışını hem de kayıtlı video dosyalarını analiz edebilir ve sürücüyü güvenilir bir şekilde belirleyebilir.  

Projenin temel amacı, video kayıtlarını veya anlık kamera görüntülerini kullanarak sürücülerin kimliğini doğrulamak ve bu verileri raporlamaktır.  

---

## 🚀 Proje Bileşenleri  

Proje üç ana Python dosyasından oluşur:  

- **yuz_kodlayici.py** → Yüz tanıma veritabanını oluşturur. **SÜRÜCÜ FOTOĞRAFLARI** klasöründeki fotoğrafları işler, yüz özelliklerini çıkarır ve **yuz_verisi_facenet.pkl** dosyasına kaydeder.  
- **canli_tanima.py** → Bilgisayar kamerasından gelen anlık video akışı üzerinden yüz tanıma yapar. Eşleşen kişileri ekranda gösterir.  
- **video_eslestirici.py** → **Videolar** klasöründeki videoları analiz eder, tespit edilen sürücüyü veritabanıyla eşleştirir ve detaylı raporu **sonuclar** klasörüne kaydeder.  

---

## ⚙️ Kurulum  

### Sistem Gereksinimleri  
- Python 3.7 veya üzeri  
- Git (Proje dosyalarını indirmek için)  
- FFmpeg (the user must download: `https://ffmpeg.org/download.html`)  

### Sanal Ortam ve Kütüphaneler  

**1. Sanal Ortam Oluşturma**  

```bash
python -m venv venv
```

**2. Sanal Ortamı Aktif Etme**  

- Windows:  
  ```bash
  .\venv\Scripts\activate
  ```

- macOS / Linux:  
  ```bash
  source venv/bin/activate
  ```

**3. Gerekli Kütüphaneleri Yükleme**  

```bash
pip install -r requirements.txt
```

> **Not:** Eğer kütüphane kurulumunda sorun yaşarsanız (özellikle büyük boyutlu `torch` nedeniyle), **yuz_kodlayici_google_colab.py** dosyasını Google Colab’da çalıştırarak `.pkl` dosyasını oluşturabilirsiniz. Daha sonra bu dosyayı indirip proje ana klasörüne koymanız yeterlidir.  

🌐 Google Colab için Çalıştırma/Google Colab’a gidin: https://colab.research.google.com/

Colab ortamında bilgisayara kurulum yapamayacağınız için **yuz_kodlayici_google_colab.py** kodunu yapıştırın ve çalıştırın.
- Tüm sürücü fotoğraflarını bir .zip dosyasında hazırlayın (örn: SÜRÜCÜ FOTOĞRAFLARI.zip).
- Aşağıdaki kodu çalıştırın, size dosya seçme penceresi açılacak.
- Çıktı olarak .pkl dosyası oluşacak, bunu indirip bilgisayarınızdaki proje klasörüne koyabilirsiniz.

Not: Colab, GPU desteği sağladığı için büyük veri ve yüz kodlamaları daha hızlı çalışır.

**Projeyi çalıştırmak için aşağıdaki Python kütüphaneleri gerekir:**

```bash
pip install torch==2.2.2 torchvision==0.17.2 facenet-pytorch==2.6.0 \
            pillow==10.2.0 opencv-python==4.12.0 numpy==1.26.4
```

---

## 📖 Kullanım Kılavuzu  

### Adım 1: Yüz Veritabanını Oluşturma  
- Sürücü fotoğraflarını **SÜRÜCÜ FOTOĞRAFLARI** klasörüne `ad_soyad.jpg` şeklinde yerleştirin.  
- Çalıştırın:  

```bash
python yuz_kodlayici.py
```

Bu işlem sonunda **yuz_verisi_facenet.pkl** otomatik olarak oluşturulacaktır.  

### Adım 2: Tanıma İşlemleri  

#### Seçenek A: Canlı Kamera ile Tanıma  

```bash
python canli_tanima.py
```

- Kameradan gelen görüntüde yüz tanıma yapılır.  
- Çıkmak için `q` tuşuna basın.  

#### Seçenek B: Video Analizi  

- `.mp4` veya `.grec` formatındaki videoları **Videolar** klasörüne yerleştirin.  
- Çalıştırın:  

```bash
python video_eslestirici.py
```

- Sonuçlar terminalde gösterilecektir.  
- Detaylı rapor **sonuclar/eslesme_raporu.txt** içinde bulunur.  

---

## 📂 Proje Yapısı  

```
/Driver-Recognition
|-- canli_tanima.py
|-- video_eslestirici.py
|-- yuz_kodlayici.py
|-- yuz_kodlayici_google_colab.py
|-- README.md
|-- README_TR.md
|-- yuz_verisi_facenet.pkl           (Oluşturulan veritabanı)
|-- ffmpeg.exe
|-- ffplay.exe                       (.exe'leri kullanıcı indirmeli)
|-- ffprobe.exe
|-- requirements.txt                 (Gerekli kütüphaneler listesi)
|-- /SÜRÜCÜ FOTOĞRAFLARI/            (Kullanıcı oluşturmalı)
|   |-- ad_soyad_1.jpg
|   |-- ad_soyad_2.png
|-- /Videolar/                       (Kullanıcı oluşturmalı)
|   |-- video_kaydi_1.mp4
|   |-- video_kaydi_2.grec
|-- /sonuclar/                       (Otomatik oluşturulur)
|-- /donusturulmus_videolar/         (Otomatik oluşturulur)
|-- /venv/                           (Sanal ortam, Kullanıcı oluşturmalı)
```

---

## ✅ Özet  

- **yuz_kodlayici.py** → Yüz veritabanı oluşturma  
- **canli_tanima.py** → Canlı kamera tanıma  
- **video_eslestirici.py** → Video eşleştirme analizi  

---

👤 **Python + Derin Öğrenme (FaceNet, MTCNN)** ile geliştirilmiştir.  
