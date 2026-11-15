"""
Streamlit demo for the power plant output model.

It trains a random forest on data/ccpp.csv the first time it loads (cached), then lets you
move the four ambient readings and see the predicted net hourly output. The full analysis,
including why the random forest wins and by how little, is in analysis.ipynb.

Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Power Plant Output", page_icon="zap", layout="centered")


@st.cache_resource
def train_model():
    df = pd.read_csv("data/ccpp.csv")
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, df


model, df = train_model()

st.title("Combined cycle power plant output")
st.caption(
    "Predicting net hourly electrical output (MW) from four ambient readings at the air "
    "intake. Trained on 9,568 hourly records from the UCI dataset. See analysis.ipynb for "
    "the full write-up."
)

st.subheader("Ambient conditions")
col1, col2 = st.columns(2)
with col1:
    at = st.slider("Ambient temperature (C)", float(df.AT.min()), float(df.AT.max()),
                   float(df.AT.mean()), 0.1)
    v = st.slider("Exhaust vacuum (cm Hg)", float(df.V.min()), float(df.V.max()),
                  float(df.V.mean()), 0.1)
with col2:
    ap = st.slider("Ambient pressure (mbar)", float(df.AP.min()), float(df.AP.max()),
                   float(df.AP.mean()), 0.1)
    rh = st.slider("Relative humidity (%)", float(df.RH.min()), float(df.RH.max()),
                   float(df.RH.mean()), 0.1)

X = pd.DataFrame([{"AT": at, "V": v, "AP": ap, "RH": rh}])
pred = float(model.predict(X)[0])

st.metric("Predicted net hourly output", f"{pred:.1f} MW")
st.caption(
    "Temperature is the dominant driver and it is negative: hot thin air gives the turbine "
    "less to work with. The observed output in this dataset ranges from about 420 to 496 MW."
)
