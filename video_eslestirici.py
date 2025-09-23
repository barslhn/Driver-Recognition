import cv2
import pickle
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
import os
import subprocess
import shutil
import time


DB_DOSYA_ADI = "yuz_verisi_facenet.pkl"

VIDEO_ANA_KLASORU = "Videolar"

DONUSTURULEN_KLASOR = "donusturulmus_videolar"

SONUC_KLASORU = "sonuclar"

FFMPEG_PATH = os.path.join(os.getcwd(), "ffmpeg.exe")

ESLESME_ESIGI = 0.9  


KONUM_AGIRLIGI = 0.7  

BUYUKLUK_AGIRLIGI = 0.3 

try:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    mtcnn = MTCNN(
        keep_all=True,
        select_largest=False,
        min_face_size=15,
        thresholds=[0.35, 0.4, 0.5],
        factor=0.6,
        device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

except Exception as e:
    print("Hata: Gerekli kütüphaneler yüklenemedi ❌")
    print(f"Detay: {e}")
    exit()
    
def yuz_veritabanini_yukle():
    """Yüz veritabanını diskten yükler."""
    print("📂 Yüz veritabanı yükleniyor...")
    try:
        with open(DB_DOSYA_ADI, "rb") as f:
            data = pickle.load(f)
        bilinen_yuz_kodlamalari = np.array(data['yuzler'])
        bilinen_yuz_isimleri = data['isimler']
        print(f"✅ Veritabanı yüklendi. {len(bilinen_yuz_isimleri)} kişi bulundu.")
        return bilinen_yuz_kodlamalari, bilinen_yuz_isimleri
    except FileNotFoundError:
        print(f"Hata: '{DB_DOSYA_ADI}' bulunamadı. Lütfen önce veritabanını oluşturun.")
        exit()

def videolari_hazirla(temizle=False):
    """
    Video klasörünü tarar ve .grec veya .mp4 uzantılı iç kamera videolarını
    analiz için .mp4 formatına dönüştürüp özel bir klasöre kopyalar.
    """
    print("\n--- Video Hazırlığı Başlıyor ---")
    if temizle and os.path.exists(DONUSTURULEN_KLASOR):
        print(f"🧹 '{DONUSTURULEN_KLASOR}' klasörü temizleniyor...")
        shutil.rmtree(DONUSTURULEN_KLASOR)
    if not os.path.exists(DONUSTURULEN_KLASOR):
        os.makedirs(DONUSTURULEN_KLASOR)

    islenen_video_sayisi = 0
    for kok_dizin, _, dosyalar in os.walk(VIDEO_ANA_KLASORU):
        for dosya_adi in dosyalar:
            if "20010100" in dosya_adi:
                continue
            
            dosya_yolu = os.path.join(kok_dizin, dosya_adi)
            goreceli_yol = os.path.relpath(kok_dizin, VIDEO_ANA_KLASORU)
            yeni_kok_dizin = os.path.join(DONUSTURULEN_KLASOR, goreceli_yol)
            if not os.path.exists(yeni_kok_dizin):
                os.makedirs(yeni_kok_dizin)
            yeni_dosya_adi = os.path.splitext(dosya_adi)[0] + ".mp4"
            yeni_dosya_yolu = os.path.join(yeni_kok_dizin, yeni_dosya_adi)

            if dosya_adi.endswith(".grec"):
                if not os.path.exists(yeni_dosya_yolu):
                    komut = [FFMPEG_PATH, '-i', dosya_yolu, yeni_dosya_yolu]
                    try:
                        subprocess.run(komut, check=True, capture_output=True, text=True)
                    except Exception as e:
                        print(f"Hata: {dosya_adi} dönüştürülürken hata: {e}")
            elif dosya_adi.endswith(".mp4"):
                if not os.path.exists(yeni_dosya_yolu):
                    shutil.copy(dosya_yolu, yeni_dosya_yolu)

            islenen_video_sayisi += 1

    if islenen_video_sayisi == 0:
        print("⚠ İç kamera videosu bulunamadı.")
        return False
    print("--- Video hazırlığı tamamlandı ✅ ---")
    return True

def videolari_analiz_et(bilinen_yuz_kodlamalari, bilinen_yuz_isimleri):
    print("\n--- Video Analizi Başlıyor ---")
    if not os.path.exists(SONUC_KLASORU):
        os.makedirs(SONUC_KLASORU)

    rapor_path = os.path.join(SONUC_KLASORU, "eslesme_raporu.txt")
    with open(rapor_path, "w", encoding='utf-8') as f:
        f.write("--- Video Analiz Raporu ---\n\n")

    for kok_dizin, _, dosyalar in os.walk(DONUSTURULEN_KLASOR):
        for video_adi in dosyalar:
            if not video_adi.endswith(".mp4") or "20010100" in video_adi:
                continue
            video_yolu = os.path.join(kok_dizin, video_adi)
            cap = cv2.VideoCapture(video_yolu)
            plaka = os.path.basename(os.path.dirname(os.path.dirname(video_yolu)))
            print(f"▶ Video: {video_adi} | Plaka: {plaka}")

            if not cap.isOpened():
                continue

            eslesen_isimler = {}
            en_yakin_mesafe = float('inf')
            frame_count = 0
            tespit_edilen_yuzler = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 10 != 0:
                    continue

                frame_height, frame_width, _ = frame.shape

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(frame_rgb)
                
                selected_face_box = None

                if boxes is not None and len(boxes) > 0:
                    tespit_edilen_yuzler += 1
                    
                    scores = []
                    for (x1, y1, x2, y2) in boxes:
                        face_area = (x2 - x1) * (y2 - y1)
                        normalized_area = face_area / (frame_width * frame_height)
                        
                        face_center_x = (x1 + x2) / 2
                        normalized_pos_x = face_center_x / frame_width
                        
                        position_score = normalized_pos_x
                        
                        total_score = (BUYUKLUK_AGIRLIGI * normalized_area) + (KONUM_AGIRLIGI * position_score)
                        scores.append(total_score)
                    
                    selected_face_index = np.argmax(scores)
                    selected_face_box = boxes[selected_face_index]
                
                if selected_face_box is not None:
                    (x1, y1, x2, y2) = selected_face_box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    yuz_crop = mtcnn.extract(frame_rgb, [selected_face_box], None)
                    
                    if yuz_crop is None:
                        print("Yüz kesiti alınamadı, atlanıyor...")
                        continue
                    if yuz_crop.dim() == 3:
                        yuz_crop = yuz_crop.unsqueeze(0)
                    
                    try:
                        kodlama = resnet(yuz_crop.to(device)).detach().cpu().numpy().squeeze()
                        mesafeler = np.linalg.norm(bilinen_yuz_kodlamalari - kodlama, axis=1)
                        current_min_distance = np.min(mesafeler)

                        if current_min_distance < en_yakin_mesafe:
                            en_yakin_mesafe = current_min_distance
                            
                        if current_min_distance < ESLESME_ESIGI:
                            isim = bilinen_yuz_isimleri[np.argmin(mesafeler)]
                            eslesen_isimler[isim] = eslesen_isimler.get(isim, 0) + 1
                    except Exception as e:
                        print(f"ResNet modelinde hata oluştu, yüz atlanıyor: {e}")
                        continue
                
                cv2.imshow("Driver Frame", frame) 
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            analiz_suresi = time.time() - start_time

            rapor = f"Plaka: {plaka}\n"
            rapor += f" > En yakın mesafe: {en_yakin_mesafe:.4f}\n"
            if tespit_edilen_yuzler == 0:
                rapor += " > Videoda hiç yüz bulunamadı ❌\n"
            elif not eslesen_isimler:
                rapor += " > Yüz bulundu ama veritabanıyla eşleşmedi ⚠\n"
            else:
                en_cok_eslesen = max(eslesen_isimler, key=eslesen_isimler.get)
                if tespit_edilen_yuzler > 0 and eslesen_isimler[en_cok_eslesen] / tespit_edilen_yuzler >= 0.5:
                    rapor += f" > Sürücü: {en_cok_eslesen} ✅\n"
                else:
                    rapor += f" > Sürücü tespit edilemedi ⚠\n"
                rapor += f" > Toplam yüz sayısı: {tespit_edilen_yuzler}\n"
                rapor += f" > Bu kişiyle eşleşme sayısı: {eslesen_isimler[en_cok_eslesen]}\n"
            rapor += f" > Analiz süresi: {analiz_suresi:.2f} sn\n\n"

            print(rapor)
            with open(rapor_path, "a", encoding='utf-8') as f:
                f.write(rapor)

    cv2.destroyAllWindows()
    print(f"--- Tüm analiz tamamlandı ✅ Rapor: {rapor_path} ---")

if __name__ == "__main__":
    bilinen_yuz_kodlamalari, bilinen_yuz_isimleri = yuz_veritabanini_yukle()
    tercih = input(f"'{DONUSTURULEN_KLASOR}' klasörünü temizleyip sıfırdan başlatmak ister misiniz? (evet/hayır): ").lower()
    temizle_klasor = True if tercih == "evet" else False
    if videolari_hazirla(temizle=temizle_klasor):
        videolari_analiz_et(bilinen_yuz_kodlamalari, bilinen_yuz_isimleri)