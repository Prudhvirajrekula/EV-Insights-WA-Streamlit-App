import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Setup
st.set_page_config(page_title="EV Insights", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚗 Washington EV Insights & Predictor</h1>", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Electric_Vehicle_Population_Data.csv")
    df = df.dropna(subset=["Make", "Model", "Base MSRP", "Electric Vehicle Type"])
    return df

df = load_data()

# Reduce dataset: Top 100 rows per Make-Model
df_filtered = df.groupby(["Make", "Model"]).head(100).reset_index(drop=True)

# Prepare training data
X = df_filtered[["Make", "Model", "Base MSRP"]]
y = df_filtered["Electric Vehicle Type"]
X_encoded = pd.get_dummies(X, columns=["Make", "Model"])

# Cache the model to avoid retraining
@st.cache_resource
def train_model(X_enc, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_enc, y)
    return model

clf = train_model(X_encoded, y)

# Optional: Show Raw Data
with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered.head())

# Chart Section
st.markdown("### 📊 EV Type Distribution in Filtered Data")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(data=df_filtered, x="Electric Vehicle Type", order=df_filtered["Electric Vehicle Type"].value_counts().index, ax=ax)
ax.set_xlabel("EV Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# User Input
st.markdown("### 🤖 Predict EV Type")
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("🔧 Select Make", sorted(df_filtered["Make"].unique()))
with col2:
    model = st.selectbox("🚗 Select Model", sorted(df_filtered[df_filtered["Make"] == make]["Model"].unique()))

msrp = st.slider("💰 Set MSRP ($)", min_value=10000, max_value=150000, step=1000, value=35000)

# Prediction Input Encoding
input_df = pd.DataFrame([[make, model, msrp]], columns=["Make", "Model", "Base MSRP"])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Predict
if st.button("🔍 Predict EV Type"):
    prediction = clf.predict(input_encoded)[0]
    st.success(f"✅ Predicted EV Type: **{prediction}**")

# Footer
st.markdown("---")
st.caption("👨‍💻 Built with ❤️ by Prudhvi Raj Rekula")
