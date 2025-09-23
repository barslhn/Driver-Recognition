import os
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

print("Fotoğrafların bulunduğu klasörü seçin...")
image_folder = filedialog.askdirectory(title="Fotoğraf klasörünü seçin")

if not image_folder:
    print("Klasör seçilmedi, işlem iptal edildi.")
    exit()

print("Kaydedilecek dosyanın kaydedileceği konumu seçin...")
save_folder = filedialog.askdirectory(title="PKL dosyasını kaydedeceğiniz klasörü seçin")

if not save_folder:
    print("Kayıt klasörü seçilmedi, işlem iptal edildi.")
    exit()

pkl_path = os.path.join(save_folder, "yuz_verisi_facenet.pkl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

face_embeddings = []
names = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(image_folder, filename)
        img = Image.open(path)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        face = mtcnn(img)
        if face is not None:
            face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu()
            face_embeddings.append(face_embedding)
            names.append(os.path.splitext(filename)[0])
            print(f"İşlendi: {filename}")
        else:
            print(f"Yüz bulunamadı: {filename}")

data = {"names": names, "embeddings": face_embeddings}
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)

print(f"\nİşlem tamamlandı. Veriler '{pkl_path}' dosyasına kaydedildi.")
