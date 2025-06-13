# 🚗 Electric Vehicle Insights – Washington State

This Streamlit app analyzes and predicts Electric Vehicle (EV) types using public EV registration data from Washington State.

## 🔍 Features

- Interactive EV type predictor using Make, Model, and MSRP
- Clean and responsive UI built with Streamlit
- Filtered dataset for fast training (top 100 rows per Make & Model)
- EV type distribution chart using Seaborn
- Cached model to ensure fast predictions

## 📁 Dataset

- Source: [WA State EV Registration Data](https://catalog.data.gov/dataset/electric-vehicle-population-data)
- Columns used: `Make`, `Model`, `Base MSRP`, `Electric Vehicle Type`

## 🚀 Run the App Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ev-insights-wa.git
cd ev-insights-wa
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

## 📷 Screenshots

![EV Predictor](assets/ev_app_demo.png)

## 👨‍💻 Author

Prudhvi Raj Rekula – MS in Computer Science  
[LinkedIn](https://www.linkedin.com/in/prudhvirajrekula) | [GitHub](https://github.com/Prudhvirajrekula)

---

**Built with ❤️ using Streamlit**
