# 🚗 EV Insights WA – Data Analysis & Prediction App

This project explores Electric Vehicle (EV) adoption trends in Washington State using public registration data. It includes:
- A Jupyter Notebook (`EV_Insights_Washington_Polished.ipynb`) for data analysis and model development
- A Streamlit web app (`app.py`) to interactively predict EV types

---

## 📊 Project Structure

### 1. **EV_Insights_Washington_Polished.ipynb**
A Jupyter Notebook containing:
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Visualization of EV trends (electric range, MSRP, location)
- Machine learning model (Random Forest) for predicting EV types
- Business recommendations based on findings

### 2. **Streamlit App (app.py)**
An interactive web application to:
- Select EV Make, Model, and Base MSRP
- Predict the EV type (BEV or PHEV)
- Visualize EV type distribution
- Built with fast, cached modeling using a filtered dataset

---

## 📁 Dataset

- Source: [Washington State EV Registration Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)
- Key Columns: `Make`, `Model`, `Base MSRP`, `Electric Vehicle Type`, `Electric Range`

---

## 💻 How to Run the Project

### ▶️ Run the Notebook
```bash
jupyter notebook EV_Insights_Washington_Polished.ipynb
```

### 🚀 Run the Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚀 Try the App (Live)
👉 [Click here to try it live](https://ev-insights-washington-prudhviraj.streamlit.app/)

---
