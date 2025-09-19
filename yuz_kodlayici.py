# Gerekli kütüphaneyi kur
!pip install facenet-pytorch

import cv2
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
import re
import numpy as np

# ZIP dosyasını çıkar
# Zip dosyasının adını tam olarak buraya yazın
!unzip "SÜRÜCÜ FOTOĞRAFLARI.zip"

# Yüz veritabanı dosyası adı
DB_DOSYA_ADI = "yuz_verisi_facenet.pkl"
# Yüz fotoğraflarının bulunduğu klasör
FOTO_KLASORU = "SÜRÜCÜ FOTOĞRAFLARI" 

# GPU varsa CUDA, yoksa CPU kullanır
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

# Yüz tespiti ve kodlaması için modelleri yükle
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("Yüz veritabanı oluşturuluyor...")
yuzler_listesi = []
surucu_isimleri_listesi = []

if not os.path.exists(FOTO_KLASORU):
    print(f"Hata: '{FOTO_KLASORU}' klasörü bulunamadı. Lütfen klasör adını kontrol edin.")
else:
    for dosya_adi in os.listdir(FOTO_KLASORU):
        if dosya_adi.endswith(('.jpg', '.jpeg', '.png')):
            dosya_yolu = os.path.join(FOTO_KLASORU, dosya_adi)
            try:
                # Dosya adından sürücü ismini çıkar (örn: "ahmet_yilmaz.jpg" -> "ahmet_yilmaz")
                surucu_adi = os.path.splitext(dosya_adi)[0]
                
                # Görüntüyü oku ve işle
                img = cv2.imread(dosya_yolu)
                if img is None:
                    print(f"Hata: {dosya_adi} okunamadı.")
                    continue
                
                # Yüzü MTCNN ile tespit et ve kes
                # MTCNN'den gelen çıktı tek bir yüz kodlaması değilse ilkini alır
                img_cropped = mtcnn(img)
                
                if img_cropped is not None:
                    # Yüz kodlamasını al
                    img_embedding = resnet(img_cropped.to(device)).detach().cpu().numpy()
                    
                    yuzler_listesi.append(img_embedding.squeeze())
                    surucu_isimleri_listesi.append(surucu_adi)
                    print(f"'{surucu_adi}' isimli sürücü fotoğrafı işlendi.")
                else:
                    print(f"Uyarı: {dosya_adi} dosyasında yüz bulunamadı.")
            except Exception as e:
                print(f"Hata: {dosya_adi} işlenirken bir sorun oluştu. Detay: {e}")

# Verileri bir pickle dosyasına kaydet
data = {'yuzler': yuzler_listesi, 'isimler': surucu_isimleri_listesi}
with open(DB_DOSYA_ADI, 'wb') as f:
    pickle.dump(data, f)

print(f"\nVeritabanı '{DB_DOSYA_ADI}' başarıyla oluşturuldu. Toplam {len(yuzler_listesi)} yüz kaydedildi.")