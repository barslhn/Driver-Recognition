# Driver Recognition Automation  

This project is an automation system that automatically detects drivers in vehicles using **Python** and deep learning libraries (**FaceNet, MTCNN**). Thanks to advanced face recognition algorithms, it can analyze both live camera streams and recorded video files and reliably identify the driver.  

The main goal of the project is to verify the identity of drivers using video recordings or live camera images and to report this data.  

---

## 🚀 Project Components  

The project consists of three main Python files:  

- **yuz_kodlayici.py** → Creates the face recognition database. It processes the photos in the **SÜRÜCÜ FOTOĞRAFLARI/DRİVER PHOTOS** folder, extracts facial features, and saves them to the **yuz_verisi_facenet.pkl** file.
- **canli_tanima.py** → Performs facial recognition on the live video stream from the computer camera. It displays the matched individuals on the screen.  
- **video_eslestirici.py** → Analyzes videos in the **Videolar/Videos** folder, matches the detected driver with the database, and saves a detailed report to the **sonuclar/results** folder.  

---

## ⚙️ Installation  

### System Requirements  
- Python 3.7 or higher  
- Git (to download project files)  
- FFmpeg (the user must download: `https://ffmpeg.org/download.html`)  
### Virtual Environment and Libraries  

**1. Creating a Virtual Environment**  

```bash
python -m venv venv
```

**2. Activating the Virtual Environment**  

- Windows:  
  ```bash
  .\venv\Scripts\activate
  ```

- macOS / Linux:  
  ```bash
  source venv/bin/activate
  ```

**3. Installing Required Libraries**  

```bash
pip install -r requirements.txt
```

> **Not:** If you encounter issues while installing libraries (especially due to the large size of `torch`), you can run the **yuz_kodlayici_google_colab.py** file in Google Colab to generate the `.pkl` file. Afterwards, simply download this file and place it in your project’s root directory.

🌐 Run on Google Colab: https://colab.research.google.com/

Since you cannot install packages directly on your computer in Colab, just paste and run the **yuz_kodlayici_google_colab.py** code.

- Prepare all driver photos in a single .zip file (e.g., DRIVER_PHOTOS.zip).
- Run the code below, and a file selection dialog will open.
- As output, a .pkl file will be generated. Download it and place it in your local project folder.

Note: Colab provides GPU support, so processing large datasets and face encodings will be faster.

**The following Python libraries are required to run the project:**

```bash
pip install torch==2.2.2 torchvision==0.17.2 facenet-pytorch==2.6.0 \
            pillow==10.2.0 opencv-python==4.12.0 numpy==1.26.4
```

---

## 📖 User Guide  

### Step 1: Creating the Face Database  
- Place the driver photos in the **SÜRÜCÜ FOTOĞRAFLARI/DRİVER PHOTOS** folder as `first_name_last_name.jpg`.  
- Run:  

```bash
python yuz_kodlayici.py
```

At the end of this process, **yuz_verisi_facenet.pkl/face_data_facenet.pkl** will be created automatically.  

### Step 2: Recognition Processes  

#### Option A: Recognition with Live Camera  

```bash
python canli_tanima.py
```

- Face recognition is performed on the image from the camera.  
- Press the `q` key to exit.  

#### Option B: Video Analysis  
- Place videos in `.mp4` or `.grec` format in the **Videolar** folder.  
- Run:  

```bash
python video_eslestirici.py
```
- Results will be displayed in the terminal.  
- A detailed report is available in **sonuclar/eslesme_raporu.txt-results/matching_report.txt**.  

---

## 📂 Project Structure  

```
/Driver-Recognition
|-- canli_tanima.py
|-- video_eslestirici.py
|-- yuz_kodlayici.py
|-- yuz_kodlayici_google_colab.py
|-- README.md
|-- README_TR.md
|-- yuz_verisi_facenet.pkl           (Created database)
|-- ffmpeg.exe
|-- ffplay.exe                       (.exe files must be downloaded by the user)
|-- ffprobe.exe
|-- requirements.txt                 (List of required libraries)
|-- /SÜRÜCÜ FOTOĞRAFLARI/            (User must create)
|   |-- first_name_last_name_1.jpg
|   |-- first_name_last_name_2.png
|-- /Videolar/                       (User must create)
|   |-- video_recording_1.mp4
|   |-- video_recording_2.grec
|-- /sonuclar/                       (Automatically created)
|-- /donusturulmus_videolar/         (Automatically created)
|-- /venv/                           (Virtual environment, User must create)
```
---

## ✅ Summary  

- **yuz_kodlayici.py** → Creating a face database  
- **canli_tanima.py** → Live camera recognition  
- **video_eslestirici.py** → Video matching analysis

---

👤 Developed with **Python + Deep Learning (FaceNet, MTCNN)**.

---

## 📝 Notes 

- Terms followed by `/tr` are the **Turkish translation** of the English term.  
- Terms followed by `-tr` are also **Turkish equivalents**, sometimes used for folder or file names.  
- This helps the reader understand folder names or context in both languages.
