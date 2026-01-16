# Football Analysis System

A comprehensive football match analysis system with player detection, action recognition, jersey number identification, and tactical visualization capabilities.

## 🎯 Features

- **Player Detection & Tracking**: Real-time player detection and tracking using YOLOv8
- **Action Recognition**: Automated recognition of football actions (passes, shots, tackles, etc.)
- **Jersey Number Recognition**: OCR-based jersey number identification using PARSeq
- **Team Assignment**: Automatic team classification based on jersey colors
- **Tactical Board**: Interactive tactical board for match analysis and visualization
- **Unsupervised Learning**: Discover new action patterns using clustering algorithms
- **Video Analysis**: Frame-by-frame analysis with action spotting

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## 🚀 Installation

### 1. Clone the repository

\\\ash
git clone https://github.com/ahmetackk/RealTime_Football_Analysis.git
cd RealTime_Football_Analysis
\\\

### 2. Install dependencies

\\\ash
pip install -r requirements.txt
\\\

### 3. Download large files

Due to GitHub's file size limitations, model files and data are hosted on Google Drive.

#### Download Models (~500MB)
[📥 Download Models Folder](https://drive.google.com/drive/folders/1r34YNyr4t_YZZuQmryFUV4VyX8xQZ7DM?usp=sharing)

Extract and place in the project root:
- \models/action_recognition.pt\
- \models/ball_detection.engine\
- \models/ball_detection.onnx\
- \models/ball_detection.pt\
- \models/pitch_detection.engine\
- \models/pitch_detection.onnx\
- \models/pitch_detection.pt\
- \models/player_detection.engine\
- \models/player_detection.onnx\
- \models/player_detection.pt\
- \jersey_recognizer/models/legibility_resnet34.pth\
- \jersey_recognizer/models/parseq_soccernet.ckpt\

#### Download Data (~600MB)
[📥 Download Data Folder](https://drive.google.com/drive/folders/1uvPGhCtWbZ_MvDCbEnw19cx_HVzfofAM?usp=sharing)

Extract and place in \unsupervised/data/\:
- \al_tactical_data.h5\
- \TAAD_sample_list.json\
- Sample videos and demo clips

## 📁 Project Structure

\\\
RealTime_Football_Analysis/
├── analyzer/              # Core analysis engines
│   ├── analyzer.py       # Main analyzer
│   └── discovery_engine.py
├── action_recognizer/     # Action recognition module
│   ├── action_recognizer.py
│   └── action_recognizer_optimized.py
├── jersey_recognizer/     # Jersey number recognition
│   ├── jersey_recognizer.py
│   └── str/parseq/       # PARSeq STR model
├── team_assigner/         # Team classification
├── tacticalboard/         # Tactical board visualization
│   ├── annotators/
│   ├── simulation/
│   └── GUI/
├── unsupervised/          # Unsupervised learning
│   ├── unsupervised.py
│   ├── models/
│   └── utils/
├── visualizer/            # Video visualization
├── gui/                   # Main GUI
├── models/                # Model files (download separately)
└── data/                  # Data files (download separately)
\\\

## 🎮 Usage

### Basic Analysis

\\\ash
python footballanalysis.py --gui
\\\

### Tactical Board

\\\ash
python tacticalboard.py
\\\

## 📊 Modules Overview

### Player Detection
Uses YOLOv8-based detection models to identify and track players on the field.

### Action Recognition
Temporal action recognition using 3D CNNs to classify football actions in video clips.

### Jersey Recognition
- **Legibility Detection**: ResNet34-based model to detect readable jersey numbers
- **OCR**: PARSeq scene text recognition for number extraction

### Team Assignment
K-means clustering on jersey colors to automatically assign players to teams.

### Tactical Analysis
- Real-time tactical board visualization
- Player position tracking
- Formation analysis
- Heat map generation

### Unsupervised Learning
- TAAD (Temporal Action Activity Discovery) baseline
- Automatic action spotting without labels
- Cluster-based action classification

## 🎓 Models Used

- **YOLOv8**: Player, ball, and pitch detection
- **PARSeq**: Jersey number OCR
- **ResNet34**: Jersey legibility classification
- **3D CNN**: Action recognition
- **TAAD**: Unsupervised action discovery

## 📝 License

This project is a graduation thesis project (2024-2025).

## 👤 Author

Ahmet Acıkök - Graduation Project

## 🙏 Acknowledgments

- SoccerNet for jersey recognition models
- Ultralytics for YOLOv8
- PARSeq team for STR models
- TAAD dataset contributors

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Note**: This is an academic project developed as a graduation thesis. Large model and data files are hosted separately due to size constraints.
