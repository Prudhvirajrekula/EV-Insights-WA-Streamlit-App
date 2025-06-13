# 🚗 Electric Vehicle Insights – Washington State

This Streamlit app analyzes and predicts Electric Vehicle (EV) types using public EV registration data from Washington State.

## 🔍 Features

- Interactive EV type predictor using Make, Model, and MSRP
- Clean and responsive UI built with Streamlit
- Filtered dataset [for live demo] for fast training (top 100 rows per Make & Model)
- EV type distribution chart using Seaborn
- Cached model to ensure fast predictions

## 📁 Dataset

- Source: [WA State EV Registration Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)
- Columns used: `Make`, `Model`, `Base MSRP`, `Electric Vehicle Type`

## 🚀 Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Prudhvirajrekula/EV-Insights-WA-Streamlit-App.git
cd EV-Insights-WA-Streamlit-App
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

## 🚀 Try the App (Live)
👉 [Click here to try it live](https://ev-insights-washington-prudhviraj.streamlit.app/)

## 👨‍💻 Author

Prudhvi Raj Rekula – MS in Computer Science  
[LinkedIn](https://www.linkedin.com/in/prudhvirajrekula) | [GitHub](https://github.com/Prudhvirajrekula)

---

**Built with ❤️ using Streamlit**
