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

> **Note:** If you encounter problems installing libraries (especially due to the large size of `torch`), you can run the **yuz_kodlayici.py** file in Google Colab to create the `.pkl` file. Then, simply download this file and place it in the project's root directory.  

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
- Place videos in `.mp4` or `.grec` format in the **Videos** folder.  
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
|-- README.md
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
- **video_eslestirici.py** → Video file analysis  

---

👤 Developed with **Python + Deep Learning (FaceNet, MTCNN)**.

---

## 📝 Notes 

- Terms followed by `/tr` are the **Turkish translation** of the English term.  
- Terms followed by `-tr` are also **Turkish equivalents**, sometimes used for folder or file names.  
- This helps the reader understand folder names or context in both languages.
