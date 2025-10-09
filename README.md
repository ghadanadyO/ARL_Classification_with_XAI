# ARL_Classification_with_XAI
## Overview

This project focuses on lung cancer detection using Reinforcement Active Learning (RAL) integrated with deep learning (CoAtNet and CNN) and traditional machine learning features.
The workflow combines feature extraction, feature selection, deep feature fusion, and Explainable AI (XAI) to assist radiologists in better understanding model predictions.

## Step 1: Dataset Preparation

Dataset Size: 790 patients with a total of 30,020 CT scan images.

Image Resolution: Each image is resized to 128 × 128 pixels for uniform input.

🤖 Step 2: Reinforcement Active Learning with CoAtNet

The dataset was divided into two parts:

Labeled data: 6,080 images

Unlabeled data: 23,940 images

The Reinforcement Active Learning (RAL) strategy was applied using a simple CoAtNet model to iteratively select informative samples for labeling, improving model accuracy with minimal human labeling effort.

🧩 Step 3: Traditional Feature Extraction and Machine Learning
Feature Extraction Techniques

Shape Features

Statistical Features

Local Binary Pattern (LBP) Features

Gray Level Run Matrix (GLRM) Features

Feature Normalization & Histogram Analysis

All extracted features were normalized and histogram-equalized to ensure consistent data distribution.

Feature Selection Methods

CFS (Correlation-based Feature Selection)

RF (Random Forest Feature Importance)

RFE (Recursive Feature Elimination)

Machine Learning Classifiers

Each selected feature subset was trained and evaluated using:

Random Forest (RF)

Decision Tree (DT)

Naïve Bayes

XGBoost

Logistic Regression

🧠 Step 4: Deep Learning with Attention Fusion
Attention Fusion of Features

Two different deep learning models were designed with attention fusion to enhance feature importance learning:

Simple CNN with Attention Fusion

Simple CoAtNet with Attention Fusion

These models aim to merge spatial and contextual information effectively.

🔗 Step 5: Combining Traditional and Deep Learning Features
Hybrid Fusion Models

Combine Traditional Features + CNN

Combine Traditional Features + CoAtNet

Combine Traditional Features + CNN with Attention Fusion Features

This hybrid fusion enhances diagnostic performance by integrating handcrafted and deep features.

🩺 Step 6: Explainable AI (XAI) with Radiologist Opinion

To ensure clinical interpretability, Explainable AI techniques were applied on the 128×128 resized CT images.
The XAI visualizations highlight regions of interest (ROI) that influence the model’s decision, aligning with radiologists’ assessments and increasing trust in model predictions.
