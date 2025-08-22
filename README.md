# Jakarta Indonesia — Sign Language Translation

This project focuses on building a system to **translate Indonesian Sign Language (BISINDO)** into text using deep learning and computer vision techniques.  
It was originally developed as part of an Omdena collaboration and is mirrored here for visibility and continued learning.

---

## 🚀 Project Overview
- **Goal**: Enable real-time or near real-time translation of BISINDO (Jakarta dialect) sign language into written text.
- **Why**: To improve accessibility and communication for the Deaf and Hard-of-Hearing community.
- **Approach**: 
  - Preprocessing of sign videos and keypoint extraction.  
  - Modeling using state-of-the-art deep learning architectures (e.g., Landmark Transformers, CNNs, LSTMs).  
  - Deployment scripts for testing inference on new video samples.  

---

## 📂 Repository Structure
├── deployment/ # Scripts for model inference and deployment
├── modeling/ # Training notebooks, experiments, and model architectures
├── preprocessing/ # Data cleaning, augmentation, and keypoint extraction pipelines
├── .gitignore
└── README.md


---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**: OpenCV, TensorFlow / PyTorch, scikit-learn, NumPy, Pandas  
- **ML Tools**: DVC, MLflow (for dataset and experiment tracking)  
- **Deployment**: Streamlit / Flask (prototype stage)

---

📊 Results (sample)

Gesture classification accuracy: 81.67% validation accuracy

Translation demo: https://lnkd.in/gU3GAjVc

📁 Data & Models

Large datasets and trained models were originally hosted on DagsHub using DVC.

Due to size limits, only sample data and scripts are provided here.

To fetch full datasets/models, please refer to the original DagsHub project.

🤝 Attribution

This repository is a mirror of the collaborative project developed through Omdena and hosted on [DagsHub](https://lnkd.in/gkrERw-h).
