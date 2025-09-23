from google.colab import files
import zipfile
import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

uploaded = files.upload()
zip_path = list(uploaded.keys())[0]

extract_dir = "/content/dataset"
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

image_extensions = ('.jpg', '.jpeg', '.png')
image_paths = []
for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.lower().endswith(image_extensions):
            image_paths.append(os.path.join(root, file))

print(f"Toplam {len(image_paths)} fotoğraf bulundu.")

for img_path in image_paths:
    try:
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)
        if face is not None:
            embedding = resnet(face.unsqueeze(0).to(device))
            print(f"✅ İşlendi: {img_path}")
        else:
            print(f"❌ Yüz bulunamadı: {img_path}")
    except Exception as e:
        print(f"Hata: {img_path} - {str(e)}")

print("✅ Tüm klasörlerdeki resimlerin işlenmesi bitti.")
