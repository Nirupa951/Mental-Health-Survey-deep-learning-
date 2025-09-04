import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# ----------------------------
# 1. Define Model Class
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # trained with BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.model(x)

# ----------------------------
# 2. Load Scaler & Model
# ----------------------------
scaler = joblib.load("scaler.pkl")

model = MLP(input_dim=3)
model.load_state_dict(torch.load("depression_model.pt"))
model.eval()

# ----------------------------
# 3. Streamlit UI
# ----------------------------
st.title("🧠 Depression Prediction App")
st.write("""
This app predicts the likelihood of **depression** based on:
- Age
- Academic Pressure
- Work Pressure

 Adjust the inputs and click **Predict** to see results.
""")

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)

with col2:
    academic = st.slider("Academic Pressure (1–10)", 1, 10, 5)

work = st.slider("Work Pressure (1–10)", 1, 10, 5)

# ----------------------------
# 4. Prediction
# ----------------------------
if st.button("Predict"):
    # Prepare input
    features = np.array([[age, academic, work]])
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        logit = model(features)
        prob = torch.sigmoid(logit).item()
        prediction = "Depressed" if prob >= 0.5 else "Not Depressed"

    # Display results
    st.subheader(f"Prediction: {prediction}")
    st.metric("Confidence", f"{prob:.2f}")
    st.progress(int(prob * 100))

# ----------------------------
# 5. Sidebar
# ----------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app was built with:
- **PyTorch** (MLP model)
- **scikit-learn** (Scaler)
- **Streamlit** (Web UI)

Model trained on a mental health survey dataset.
""")