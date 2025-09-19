import cv2
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

# FaceNet modellerini yükle
# GPU varsa CUDA, yoksa CPU kullanır
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Yüz tespiti için MTCNN modelini yükle
# `keep_all=True` ile karedeki tüm yüzleri algılar
mtcnn = MTCNN(keep_all=True, device=device)
# Yüz kodlaması için InceptionResnetV1 modelini yükle
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Yüz veritabanı yükleniyor...")
try:
    with open("yuz_verisi_facenet.pkl", "rb") as f:
        data = pickle.load(f)
    bilinen_yuz_kodlamalari = data['yuzler']
    bilinen_yuz_isimleri = data['isimler']
    print("Veritabanı başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'yuz_verisi_facenet.pkl' dosyası bulunamadı. Lütfen 'yuz_kodlayici.py' dosyasını çalıştırın.")
    exit()
except Exception as e:
    print(f"Hata: Veritabanı yüklenirken bir sorun oluştu. Detay: {e}")
    exit()

# Eşleşme için bir eşik değeri belirle (daha düşük değerler daha iyi eşleşme gerektirir)
ESLEME_ESIGI = 0.8

# Kamera akışını başlat
video_yakalama = cv2.VideoCapture(0)

while True:
    ret, frame = video_yakalama.read()
    if not ret:
        print("Kamera akışı okunamadı. Çıkış yapılıyor.")
        break
    
    # Yüzleri MTCNN ile tespit et
    # `mtcnn.detect` yüzlerin kutu koordinatlarını ve hassasiyetlerini döndürür
    boxes, _ = mtcnn.detect(frame)
    
    if boxes is not None:
        # Tespit edilen yüzlerin her birinin koordinatlarını al
        yuzlerin_koordinatlari = boxes.astype(int)
        
        # Yüzleri resimden ayır ve kodlamalarını al
        yuz_goruntuleri = mtcnn.extract(frame, boxes, None)
        yuz_kodlamalari_live = resnet(yuz_goruntuleri.to(device)).detach().cpu().numpy()

        for i, kodlama in enumerate(yuz_kodlamalari_live):
            isim = "Bilinmeyen"
            
            # Kayıtlı yüzlerle karşılaştır
            mesafeler = np.linalg.norm(np.array(bilinen_yuz_kodlamalari) - kodlama, axis=1)
            en_yakin_mesafe = np.min(mesafeler)
            en_yakin_index = np.argmin(mesafeler)
            
            if en_yakin_mesafe < ESLEME_ESIGI:
                isim = bilinen_yuz_isimleri[en_yakin_index]

            # Yüzün etrafına kutu çiz
            x1, y1, x2, y2 = yuzlerin_koordinatlari[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, isim, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Görüntüyü ekranda göster
    cv2.imshow('Sürücü Tanıma Otomasyonu', frame)
    
    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temiz bir şekilde çıkış yap
video_yakalama.release()
cv2.destroyAllWindows()