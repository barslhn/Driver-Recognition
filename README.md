# Driver Recognition Automation  

This project is an automation system that automatically detects drivers in vehicles using **Python** and deep learning libraries (**FaceNet, MTCNN**). Thanks to advanced face recognition algorithms, it can analyze both live camera streams and recorded video files and reliably identify the driver.  

The main goal of the project is to verify the identity of drivers using video recordings or live camera images and to report this data.  

---

## 🚀 Project Components  

The project consists of three main Python files:  

- **yuz_kodlayici.py** → Creates the face recognition database. It processes the photos in the **DRIVER PHOTOS** folder, extracts facial features, and saves them to the **yuz_verisi_facenet.pkl** file.
- **canli_tanima.py** → Performs facial recognition on the live video stream from the computer camera. It displays the matched individuals on the screen.  
- **video_eslestirici.py** → Analyzes videos in the **Videos** folder, matches the detected driver with the database, and saves a detailed report to the **results** folder.  

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
- Place the driver photos in the **DRIVER PHOTOS** folder as `first_name_last_name.jpg`.  
- Run:  

```bash
python face_encoder.py
```

At the end of this process, **face_data_facenet.pkl** will be created automatically.  

### Step 2: Recognition Processes  

#### Option A: Recognition with Live Camera  

```bash
python live_recognition.py
```

- Face recognition is performed on the image from the camera.  
- Press the `q` key to exit.  

#### Option B: Video Analysis  
- Place videos in `.mp4` or `.grec` format in the **Videos** folder.  
- Run:  

```bash
python video_matcher.py
```
- Results will be displayed in the terminal.  
- A detailed report is available in **results/matching_report.txt**.  

---

## 📂 Project Structure  

```
/Driver-Recognition
|-- live_recognition.py
|-- video_matcher.py
|-- face_encoder.py
|-- README.md
|-- face_data_facenet.pkl           (Created database)
|-- ffmpeg.exe
|-- ffplay.exe                       (.exe files must be downloaded by the user)
|-- ffprobe.exe
|-- requirements.txt                 (List of required libraries)
|-- /DRIVER PHOTOS/            (User must create)
|   |-- first_name_last_name_1.jpg
|   |-- first_name_last_name_2.png
|-- /Videos/                       (User must create)
|   |-- video_recording_1.mp4
|   |-- video_recording_2.grec
|-- /results/                       (Automatically created)
|-- /converted_videos/         (Automatically created)
|-- /venv/                           (Virtual environment, User must create)
```
---

## ✅ Summary  

- **face_encoder.py** → Creating a face database  
- **live_recognition.py** → Live camera recognition  
- **video_matcher.py** → Video file analysis  

---

👤 Developed with **Python + Deep Learning (FaceNet, MTCNN)**.