import cv2
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
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

ESLEME_ESIGI = 0.8

video_yakalama = cv2.VideoCapture(0)

while True:
    ret, frame = video_yakalama.read()
    if not ret:
        print("Kamera akışı okunamadı. Çıkış yapılıyor.")
        break
    
    boxes, _ = mtcnn.detect(frame)
    
    if boxes is not None:
        yuzlerin_koordinatlari = boxes.astype(int)
        
        yuz_goruntuleri = mtcnn.extract(frame, boxes, None)
        yuz_kodlamalari_live = resnet(yuz_goruntuleri.to(device)).detach().cpu().numpy()

        for i, kodlama in enumerate(yuz_kodlamalari_live):
            isim = "Bilinmeyen"
            
            mesafeler = np.linalg.norm(np.array(bilinen_yuz_kodlamalari) - kodlama, axis=1)
            en_yakin_mesafe = np.min(mesafeler)
            en_yakin_index = np.argmin(mesafeler)
            
            if en_yakin_mesafe < ESLEME_ESIGI:
                isim = bilinen_yuz_isimleri[en_yakin_index]

            x1, y1, x2, y2 = yuzlerin_koordinatlari[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, isim, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Sürücü Tanıma Otomasyonu', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_yakalama.release()
cv2.destroyAllWindows()