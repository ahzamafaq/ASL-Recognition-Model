# ASL Recognition using Multi-Modal Deep Learning

## 📌 Project Overview
This project implements an **American Sign Language (ASL) recognition system** using **deep learning**. It includes three approaches:

1. **Baseline Model:** Image-based CNN for ASL recognition.
2. **Skeletal Model:** Uses **Mediapipe Hands** to extract hand landmarks and classify ASL gestures.
3. **Multi-Modal Model:** Combines both image and skeletal data for enhanced recognition accuracy.

The models are trained on the **ASL Alphabet Dataset** from **Kaggle**:  
[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  

---

# 📦 Project Name

## 📂 Final_Codes/
- 📜 `baseline_do_not_touch.py` - Baseline model execution
- 📜 `skeletal_do_not_touch.py` - Skeletal model execution
- 📜 `multi_modal_do_not_touch.py` - Multi-modal model execution
- 📜 `final3_combined.py` - Final integrated model

## 📂 keypoints_csv_files/
- 📜 `skeletal_keypoints.csv` - CSV with extracted keypoints

## 📂 model_files/
### 📁 Baseline/
- 📜 `preprocess_asl.py` - Data augmentation & preprocessing
- 📜 `splitting.py` - Dataset splitting
- 📜 `model_training.py` - CNN Model training

### 📁 Multi_Modal/
- 📜 `multi_modal_preprocessing.py` - Feature extraction
- 📜 `multi_modal_training.py` - Multi-modal model training

### 📁 Skeletal/
- 📜 `skeletal_extraction.py` - Extracts skeletal keypoints
- 📜 `skeletal_training.py` - Skeletal model training

## 📂 Models/
- 📜 `baseline_model.h5` - CNN model
- 📜 `multimodal_model_large_non_augmented.h5` - Multi-modal model
- 📜 `skeletal_model_large_non_augmented.h5` - Skeletal model

## 📂 asl_dataset/
- ASL dataset (from Kaggle)
