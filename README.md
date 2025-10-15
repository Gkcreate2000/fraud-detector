# 🚨 Unified Fraud/Spam/Phishing Detector

A machine learning web application built with Streamlit that detects suspicious messages, SMS, and URLs in real-time.

## 🔍 Overview

This application uses machine learning to classify text messages as either legitimate or potentially fraudulent/spam/phishing attempts. It provides:

- **Real-time Detection**: Analyze individual messages or URLs instantly
- **Model Training**: Upload CSV datasets to train and improve the detection model
- **Customizable Interface**: Choose between Dark and RGB Neon themes
- **Confidence Scoring**: Get probability scores for each prediction

## 🛠️ Features

- **Text Analysis**: Enter any SMS, message, or URL to check for suspicious content
- **CSV Training**: Upload labeled datasets to train custom models
- **Live Predictions**: Instant classification with confidence percentages
- **Theme Support**: Dark mode and subtle RGB Neon themes
- **Model Persistence**: Automatically saves and loads trained models

## 📊 How It Works

1. **Input**: User provides text message or URL
2. **Processing**: TF-IDF vectorization converts text to numerical features
3. **Classification**: Logistic Regression model predicts fraud/spam probability
4. **Output**: Returns classification with confidence scores

## 🎯 Use Cases

- Detect phishing attempts in emails and messages
- Identify spam SMS and social media messages
- Analyze suspicious URLs and links
- Train custom models for specific fraud patterns

## 🏗️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, TF-IDF, Logistic Regression
- **Data Processing**: pandas, numpy
- **Model Storage**: joblib

## 📁 Project Structure
