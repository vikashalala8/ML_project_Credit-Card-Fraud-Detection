import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(135deg,#0F172A,#1E3A8A,#2563EB);
}

/* Title */
.title{
    font-size:55px;
    font-weight:bold;
    text-align:center;
    color:#00F5FF;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:22px;
    margin-bottom:30px;
}

/* Cards */
.card{
    background:white;
    padding:20px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

/* Upload Box */
[data-testid="stFileUploader"]{
    background:#FFFFFF20;
    border:2px dashed #00F5FF;
    border-radius:15px;
    padding:20px;
}

/* Button */
.stButton>button{
    background:linear-gradient(to right,#06B6D4,#2563EB);
    color:white;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    height:50px;
}

.stDownloadButton>button{
    background:#10B981;
    color:white;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

h3{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------

st.markdown(
"""
<div class="title">💳 Credit Card Fraud Detection</div>
<div class="subtitle">
Detect fraudulent credit card transactions using Machine Learning
</div>
""",
unsafe_allow_html=True
)

# ------------------ SIDEBAR ------------------

st.sidebar.title("ℹ Project Information")

st.sidebar.info("""
**Model Used**

✔ Logistic Regression

**Dataset**

✔ Credit Card Fraud Detection Dataset

**Features**

✔ Upload CSV

✔ Fraud Prediction

✔ Download Results
""")

# ------------------ FILE UPLOAD ------------------

uploaded_file = st.file_uploader(
    "📂 Upload Credit Card CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully!")

    st.markdown("### 📋 Uploaded Dataset")

    st.dataframe(
        data.head(),
        use_container_width=True
    )

    # ------------------ PREPARE DATA ------------------

    if "Class" in data.columns:
        X = data.drop("Class", axis=1)
    else:
        X = data

    prediction = model.predict(X)

    data["Prediction"] = prediction

    data["Prediction"] = data["Prediction"].map(
        {
            0:"Legitimate",
            1:"Fraud"
        }
    )

    fraud = (data["Prediction"]=="Fraud").sum()
    legit = (data["Prediction"]=="Legitimate").sum()
    total = len(data)

    # ------------------ METRICS ------------------

    st.markdown("## 📊 Dashboard")

    c1,c2,c3 = st.columns(3)

    c1.markdown(
        f"""
        <div class="card">
        <h2>📁 Total</h2>
        <h1>{total}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    c2.markdown(
        f"""
        <div class="card">
        <h2>✅ Legitimate</h2>
        <h1 style="color:green;">{legit}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    c3.markdown(
        f"""
        <div class="card">
        <h2>🚨 Fraud</h2>
        <h1 style="color:red;">{fraud}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ------------------ CHARTS ------------------

    col1,col2 = st.columns(2)

    with col1:

        fig = px.pie(
            names=["Legitimate","Fraud"],
            values=[legit,fraud],
            color=["Legitimate","Fraud"],
            color_discrete_sequence=[
                "#22C55E",
                "#EF4444"
            ],
            title="Prediction Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.bar(
            x=["Legitimate","Fraud"],
            y=[legit,fraud],
            color=["Legitimate","Fraud"],
            color_discrete_sequence=[
                "#22C55E",
                "#EF4444"
            ],
            title="Prediction Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    # ------------------ RESULTS ------------------

    st.markdown("# 📄 Prediction Results")

    st.dataframe(
        data,
        use_container_width=True,
        height=500
    )

    # ------------------ DOWNLOAD ------------------

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction CSV",
        csv,
        "prediction_results.csv",
        "text/csv"
    )

# ------------------ FOOTER ------------------

st.markdown("---")

st.markdown(
"""
<center>

### 💻 Developed using Streamlit

Machine Learning Project • Credit Card Fraud Detection

</center>
""",
unsafe_allow_html=True
)