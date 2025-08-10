import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Define SensorNet (same as in your training) ---
class SensorNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
        super(SensorNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(output_dim, 3)  # 3 classes

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

# --- Define RipenessClassifier (EfficientNet) ---
class RipenessClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(RipenessClassifier, self).__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b3')
        num_features = self.base_model._fc.in_features
        self.base_model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# --- Late Fusion Model ---
class LateFusionModel(nn.Module):
    def __init__(self, cnn_model, sensor_model, w_cnn=0.6, w_sensor=0.4):
        super(LateFusionModel, self).__init__()
        self.cnn_model = cnn_model
        self.sensor_model = sensor_model
        self.w_cnn = w_cnn
        self.w_sensor = w_sensor

    def forward(self, image, sensor):
        cnn_probs = F.softmax(self.cnn_model(image), dim=1)
        sensor_probs = F.softmax(self.sensor_model(sensor), dim=1)
        fused_probs = self.w_cnn * cnn_probs + self.w_sensor * sensor_probs
        return fused_probs

# --- Load Models ---
num_classes = 3
banana_cnn_model = RipenessClassifier(num_classes).to(device)
banana_sensor_model = SensorNet().to(device)

# # Load the model
# model = torch.load('best_banana_model.pth', map_location='cpu')
# model.eval()  # Set the model to evaluation mode

# Load weights (adjust paths if needed)
cnn_checkpoint = torch.load('best_banana_model.pth', map_location='cpu')
banana_cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
banana_cnn_model.eval()

sensor_state_dict = torch.load('banana_900_sensor_model.pth', map_location=device)
banana_sensor_model.load_state_dict(sensor_state_dict)
banana_sensor_model.eval()

fusion_model = LateFusionModel(banana_cnn_model, banana_sensor_model, w_cnn=0.6, w_sensor=0.4).to(device)
fusion_model.eval()

# --- Load scaler for sensor data ---
scaler = joblib.load('banana_sensor_scaler.pkl')  # Make sure this file exists

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("Banana Ripeness Predictor (Late Fusion)")

uploaded_file = st.file_uploader("Upload an image of the banana:", type=["jpg", "jpeg", "png"])

mq4 = st.number_input("Sensor MQ4 value:", min_value=0.0, max_value=100.0, value=0.0)
mq135 = st.number_input("Sensor MQ135 value:", min_value=0.0, max_value=100.0, value=0.0)
tgs2602 = st.number_input("Sensor TGS2602 value:", min_value=0.0, max_value=100.0, value=0.0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict Ripeness"):
        # Preprocess image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Preprocess sensor
        sensor_vals = np.array([mq4, mq135, tgs2602]).reshape(1, -1)
        sensor_vals = sensor_vals * np.array([0.2, 0.6, 0.2])  # Apply weights same as training
        sensor_scaled = scaler.transform(sensor_vals)
        sensor_tensor = torch.tensor(sensor_scaled, dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            probs = fusion_model(img_tensor, sensor_tensor)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        label_map = {0: "Ripe", 1: "Rotten", 2: "Unripe"}
        st.write(f"### Predicted Ripeness: {label_map[pred_class]}")
        st.write(f"Confidence: {confidence:.2f}")

