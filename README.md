# Football Analysis System

A comprehensive football match analysis system with player detection, action recognition, jersey number identification, and tactical visualization.

## Features

- Player Detection & Tracking (YOLOv8)
- Action Recognition (3D CNN)
- Jersey Number Recognition (PARSeq + ResNet34)
- Team Assignment
- Tactical Board Visualization
- Unsupervised Action Discovery

## Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

## Installation
```bash
git clone https://github.com/ahmetackk/RealTime_Football_Analysis.git
cd RealTime_Football_Analysis
pip install -r requirements.txt
```

## Download Large Files

### Models (~500MB)
[Download Models](https://drive.google.com/drive/folders/1r34YNyr4t_YZZuQmryFUV4VyX8xQZ7DM?usp=sharing)

Place in project root as shown in structure below.

### Data (~600MB)
[Download Data](https://drive.google.com/drive/folders/1uvPGhCtWbZ_MvDCbEnw19cx_HVzfofAM?usp=sharing)

Place in `unsupervised/data/` directory.

## Project Structure
```
RealTime_Football_Analysis/
│
├── analyzer/
├── action_recognizer/
├── jersey_recognizer/
├── team_assigner/
├── tacticalboard/
├── unsupervised/
├── visualizer/
├── gui/
│
├── models/                    # Download from Google Drive
│   ├── action_recognition.pt
│   ├── player_detection.onnx
│   ├── ball_detection.onnx
│   └── pitch_detection.onnx
│
└── data/                      # Download from Google Drive
```

## Usage
```bash
# Basic analysis
python footballanalysis.py

# GUI
python gui/football_analysis_gui.py

# Tactical board
python tacticalboard.py

# Unsupervised analysis
python unsupervised_analyzer.py
```

## Models

- YOLOv8: Player, ball, and pitch detection
- PARSeq: Jersey number OCR
- ResNet34: Jersey legibility classification
- 3D CNN: Action recognition
- TAAD: Unsupervised action discovery

## Author

Ahmet Acıkök - Graduation Project (2024-2025)

## Acknowledgments

SoccerNet, Ultralytics, PARSeq team, TAAD contributors
