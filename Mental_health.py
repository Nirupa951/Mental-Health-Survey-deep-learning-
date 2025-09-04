import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv(
    r'C:\Users\DELL\Desktop\nirupa\Guvi\Miniproject5_mental_health_survey\playground-series-s4e11/train.csv'
)

# Ensure labels are 0/1
if df["Depression"].dtype == "object":
    df["Depression"] = df["Depression"].map({"No": 0, "Yes": 1})
else:
    df["Depression"] = df["Depression"].replace({2: 1, -1: 0})

print("Unique labels in Depression column:", df["Depression"].unique())

# ----------------------------
# 2. Select features
# ----------------------------
features = ["Age", "Academic Pressure", "Work Pressure"]

#  Handle NaNs (fill with column mean)
df[features] = df[features].fillna(df[features].mean())

X = df[features].values
y = df["Depression"].values

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Scale features
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print(" Saved scaler.pkl")

# Debug check for NaNs
print("NaN in X_train:", np.isnan(X_train).any())
print("NaN in y_train:", np.isnan(y_train).any())

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ----------------------------
# 5. Define MLP
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)   # no sigmoid here
        )
    def forward(self, x):
        return self.model(x)

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ----------------------------
# 6. Train model
# ----------------------------
model = MLP(input_dim=3)
model.apply(init_weights)

criterion = nn.BCEWithLogitsLoss()   #  stable BCE
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # smaller LR

num_epochs = 50
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----------------------------
# 7. Evaluate model
# ----------------------------
with torch.no_grad():
    logits = model(X_test)                  # raw logits
    probs = torch.sigmoid(logits)           # convert to probabilities
    y_pred_class = (probs >= 0.5).int()     # threshold
    accuracy = (y_pred_class.eq(y_test).sum().item()) / len(y_test)
    print(f" Test Accuracy: {accuracy:.4f}")

# ----------------------------
# 8. Save trained model
# ----------------------------
torch.save(model.state_dict(), "depression_model.pt")
print("Model saved as depression_model.pt")