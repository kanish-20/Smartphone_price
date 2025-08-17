# Smartphone Price Prediction

This project predicts the price of a smartphone based on its specifications such as brand, model, RAM, storage, and 5G support.

The backend is built with Flask, the model is trained using **ElasticNet Regression**, and the frontend is a simple HTML/CSS form.

## Features

**Predict smartphone prices based on:**

- Brand

- Model

- RAM (GB)

- Storage (GB)

- 5G Support (Yes/No)

- Uses ElasticNet Regression for balancing L1 (Lasso) and L2 (Ridge) regularization.

- Interactive web interface with Flask.

## Tech Stack

- Python 3.8+

- Flask – Backend web framework

- Scikit-learn – Model training (ElasticNet Regression)

- Pandas, NumPy – Data preprocessing

- HTML, CSS – Frontend

## Project Structure

```
Smartphone_Price_Prediction/
│── train_model.py        # Script to train the ElasticNet model
│── app.py                # Flask application
│── smartphone_price_model.pkl  # Saved trained model
│── templates/
│   └── index.html        # Frontend HTML form
│── static/
│   └── style.css         # Optional extra CSS (if separated)
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

## Installation

### 1. Clone the repository or download the project
```
git clone https://github.com/yourusername/smartphone-price-prediction.git
cd smartphone-price-prediction
```

### 2. Install dependencies.
```
pip install -r requirements.txt
```

### 3. Training the Model
```
python train_model.py
```

### 4. Run the web app
```
python app.py
```

## Algorithm

We are using ElasticNet Regression:

- L1 Regularization (Lasso) → helps in feature selection by shrinking irrelevant features to zero.

- L2 Regularization (Ridge) → reduces overfitting by penalizing large coefficients.

- ElasticNet combines both for better performance on high-dimensional data.

## Screenshots

![WhatsApp Image 2025-08-17 at 10 09 53](https://github.com/user-attachments/assets/66864533-30e2-4cf7-b2e2-c7cc70e6bc5c)
![WhatsApp Image 2025-08-17 at 10 09 53 (1)](https://github.com/user-attachments/assets/0dc3a475-9a00-41df-bebd-755c6b47176b)

