# Food Nut Advanced Chatbot 🤖🍲

A smart AI-powered chatbot and food analysis system that combines image-based food classification, freshness prediction, nutritional analysis, and interactive chatbot functionalities. Built using deep learning, NLP, and Streamlit for a user-friendly interface.

---

## Table of Contents
1. [Dataset and Preprocessing](#dataset-and-preprocessing)
2. [Algorithms and Models Used](#algorithms-and-models-used)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Application Workflow](#application-workflow)
5. [Technologies and Libraries](#technologies-and-libraries)
6. [Features and Innovations](#features-and-innovations)
7. [Key Challenges and Solutions](#key-challenges-and-solutions)
8. [Setup Instructions](#setup-instructions)
9. [Usage](#usage)
10. [Contact](#contact)

---

## Dataset and Preprocessing

### Dataset Used
- **Food-101 Dataset**: 101 food categories with 101,000 images (1,000 per class).  
- **Nutritional Dataset**: Contains calorie and nutrient information, merged with Food-101 categories.

### Preprocessing Steps
1. **Image Preprocessing**:  
   - Resized to 224x224 pixels  
   - Normalized pixel values to [0, 1]  

2. **Augmentation**:  
   - Random rotation, horizontal flipping, zooming, brightness adjustments  

3. **Data Splitting**:  
   - 70% training, 20% validation, 10% testing  

4. **Label Encoding**:  
   - Food categories → numerical labels  
   - Nutritional values scaled using `MinMaxScaler` for regression

---

## Algorithms and Models Used

### a. Food Classification Model
- **Model**: EfficientNetB0 (pre-trained on ImageNet)  
- **Reason**: High accuracy with low computational cost  
- **Architecture**:  
  - Base EfficientNetB0 for feature extraction  
  - Custom fully connected layers for 101-class classification  
  - Output Layer: Softmax activation  

### b. Freshness Prediction
- **Model**: Custom CNN  
- **Architecture**:  
  - 3 convolutional layers + ReLU  
  - MaxPooling and Dropout  
  - Sigmoid output layer for binary classification (Fresh / Not Fresh)

### c. Calorie and Nutritional Estimation
- **Approach**:  
  - Features extracted using EfficientNetB0  
  - Linear Regression model predicts calories and nutrients (protein, fat, carbs)  

### d. Chatbot with NLP
- **Model**: BERT-based Hugging Face model for intent recognition  
- **Capabilities**:  
  - Understands user queries related to food  
  - Provides insights like:  
    - "Is this food suitable for children?"  
    - "What are alternatives for this food?"

---

## Model Training and Evaluation

### Training Steps
1. Initialized EfficientNetB0 with frozen layers, then fine-tuned  
2. Trained freshness model on manually labeled data  
3. Trained calorie estimation regression on extracted features  

### Training Details
- Optimizer: Adam  
- Loss Functions:  
  - Categorical Cross-Entropy (classification)  
  - Binary Cross-Entropy (freshness)  
  - Mean Squared Error (calorie/nutrition)  
- Batch Size: 32  
- Epochs: 20  

### Evaluation Metrics
- Classification Accuracy (food identification)  
- Precision, Recall, F1-Score (freshness prediction)  
- Mean Absolute Error (calorie/nutrition regression)

---

## Application Workflow

1. **User Input**: Upload food image via Streamlit interface  
2. **Food Identification**: EfficientNetB0 predicts food category with confidence score  
3. **Freshness Prediction**: Custom CNN predicts Fresh / Not Fresh  
4. **Nutritional Analysis**: Regression model predicts calories and nutrient breakdown  
5. **Chatbot Interaction**: User queries handled by BERT-based NLP model  
6. **Output Display**:  
   - Predicted food category  
   - Freshness status  
   - Nutritional values and calorie count  
   - Dietary advice

---

## Technologies and Libraries

- **Deep Learning**: TensorFlow, Keras  
- **NLP**: Hugging Face Transformers  
- **Frontend**: Streamlit  
- **Image Processing**: OpenCV, Pillow  
- **Data Handling**: NumPy, Pandas, Scikit-learn  
- **Visualization**: Matplotlib

---

## Features and Innovations

1. **Integrated Solution**: Image-based food identification + freshness + NLP chatbot  
2. **User-Friendly Interface**: Interactive Streamlit dashboard with real-time predictions  
3. **Multi-Model Approach**: EfficientNetB0, Custom CNN, Regression for nutrition estimation  

---

## Key Challenges and Solutions

- **Freshness Labels Missing** → Manually labeled subset for training  
- **Complex User Queries** → Fine-tuned BERT for intent recognition  
- **Overfitting** → Data augmentation, dropout, and early stopping  

---

## Setup Instructions

1. **Clone the repo**:

```bash
git clone https://github.com/Vishand1403/food-nut-adv-chatbot.git
cd food-nut-adv-chatbot
