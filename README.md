# 📈 SmartStock-ML-Predictor

### Intelligent Stock Market Analysis using Machine Learning (No Deep Learning)

A powerful **Machine Learning-based web application** that analyzes stock market data, trains **17 ML models (classification + regression)**, and automatically selects the best model to predict stock price and trend with high accuracy.

---

## 🚀 Features

✅ 17 Machine Learning Models (9 Classification + 8 Regression)
✅ Automatic Best Model Selection
✅ Real-time Stock Data using yFinance
✅ 30+ Advanced Feature Engineering
✅ Clean & Interactive UI using Streamlit
✅ UP / DOWN Prediction with Confidence
✅ Confusion Matrix & Performance Metrics
✅ Graphs & Visual Insights
✅ Save Trained Models

---

## 🧠 Model System

### 📊 Classification (Trend Prediction)

Predicts whether stock will go **UP 📈 or DOWN 📉**

* Logistic Regression
* Decision Tree
* Random Forest
* KNN
* SVM
* Naive Bayes
* AdaBoost
* Gradient Boosting
* XGBoost

---

### 📉 Regression (Price Prediction)

Predicts **future stock price**

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* KNN Regressor
* SVR
* AdaBoost Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

## 🏆 Smart Model Selection

* **Classification → F1 Score**
* **Regression → R² Score**

Automatically selects the **best performing model** 🔥

---

## 📁 Project Structure

```bash
stock_ml_project/
│
├── app.py
├── train.py
├── evaluation.py
├── run_pipeline.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── data_fetcher.py
│   ├── visualizer.py
│   └── helpers.py
│
├── outputs/        # Screenshots (used in README)
├── models/
├── data/
├── plots/
└── results/
```

---

## 🧪 Installation & Setup

### 📥 Step 1: Clone Repository

```bash
git clone https://github.com/your-username/SmartStock-ML-Predictor.git
cd SmartStock-ML-Predictor
```

### 🧰 Step 2: Create Virtual Environment

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run Application

```bash
streamlit run app.py
```

Open 👉 http://localhost:8501

---

## 📸 Application Screenshots

### 🖥️ UI Interface

![UI](outputs/ui.png)

---

### 🤖 Model Training Dashboard

![Models](outputs/models.png)

---

### 📥 Data Fetching

![Fetch](outputs/fetch.png)

---

### 🔮 Prediction Output

![Prediction](outputs/prediction.png)

![Prediction1](outputs/prediction1.png)

---

### 📊 Classification Results

![Classification](outputs/classification.png)

---

### 📉 Regression Results

![Regression](outputs/regg.png)

---

### 📌 Confusion Matrix

![Confusion](outputs/confusion.png)

---

### 🧠 All Models Comparison

![All Models](outputs/allmodels.png)

---

### 📈 Charts & Visualizations

![Charts](outputs/charts.png)

---

### 💾 Save Model Option

![Save](outputs/save.png)


## 📊 Features Engineered

* Moving Averages (SMA, EMA)
* RSI Indicator
* MACD
* Bollinger Bands
* Volatility Metrics
* Momentum Indicators
* Volume Analysis
* Lag Features

---

## 📏 Evaluation Metrics

### Classification

* Accuracy
* Precision
* Recall
* F1 Score

### Regression

* MAE
* MSE
* RMSE
* R² Score


## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
