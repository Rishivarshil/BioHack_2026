import sys
import subprocess
import time
from tkinter import Image
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import joblib
import joblib
import pandas as pd
import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
import numpy as np

img_path = ""

class DragDropArea(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image to Covert")
        self.resize(400, 200)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()
        self.label = QLabel("Drag and Drop Image Here", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            border: 2px dashed #aaa;
            border-radius: 10px;
            font-size: 16px;
            background-color: #f9f9f9;
        """)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        # Check if the dropped object is a file
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            image_path = files[0]
            self.label.setText(f"Processing:\n{image_path}")
            self.run_covert_script(image_path)

    def run_covert_script(self, path):
        try:
            # Executes: python covert.py "C:/path/to/image.png"
            img_path = path
            subprocess.run(["python", "train.py", path], check=True)
            print(f"Successfully sent {path} to train.py")
        except subprocess.CalledProcessError as e:
            print(f"Error running train.py: {e}")
        except FileNotFoundError:
            print("Error: 'covert.py' not found in the current directory.")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(base.children())[:-1]) 
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        features = self.encoder(x)
        vector = self.flatten(features)
        return vector 

def strict_medical_crop(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        x, y, w, h = cv2.boundingRect(c)
        
        cropped_img = img_bgr[y:y+h, x:x+w]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        c_shifted = c - [x, y]
        cv2.drawContours(mask, [c_shifted], -1, 255, -1)
        final_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
    else:
        final_img = img_bgr

    return Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))

def visualize_grad_cam(model):
    model = models.resnet18(weights='IMAGENET1K_V1')
    target_layers = [model.layer4[-1]] 

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(281)] 

    grayscale_cam = cam(input_tensor=input_batch, targets=targets)
    grayscale_cam = grayscale_cam[0, :] 


    img_float =   np.array(img).astype(np.float32) / 255.0
    img_float = cv2.resize(img_float, (224, 224))
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    cv2.imwrite('grad_cam_result.jpg', visualization[:, :, ::-1])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
categories = {"kidney": 0, "tumor": 1} 


if True:
    img = Image.open(image_path).convert('RGB')
else:
    img = strict_medical_crop(image_path)





input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) 

model = FeatureExtractor()
model.eval() 

with torch.no_grad():
    vector_embedding = model(input_batch)

print(f"Success! Your .jpg is now a vector of shape: {vector_embedding.shape}")
df = pd.DataFrame(vector_embedding.cpu().numpy(), columns=[f"dim_{i}" for i in range(vector_embedding.shape[1])])
df.to_csv('vector_.csv', index=False)

start_time_pre = time.time()

class KidneyAutoencoder(nn.Module):
    def __init__(self):
        super(KidneyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64) 
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class KidneyClassifier(nn.Module):
    def __init__(self, trained_encoder):
        super(KidneyClassifier, self).__init__()
        self.encoder = trained_encoder
        
        self.classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier_head(features)



scaler = joblib.load('data_scaler.pkl')

temp_ae = KidneyAutoencoder() 
model = KidneyClassifier(temp_ae.encoder)

model.load_state_dict(torch.load('kidney_classifier.pth'))
model.eval()

def predict_new_data(csv_path):
    new_df = pd.read_csv(csv_path)
    
    features = new_df.filter(regex='^dim_').values
    features_scaled = scaler.transform(features)
    input_tensor = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        raw_probs = model(input_tensor)
        predictions = (raw_probs > 0.5).int()
        

        latent_features = model.encoder(input_tensor)
    
    return predictions.flatten(), latent_features

# Usage
labels, vectors = predict_new_data('vector_.csv')
if labels[0] == 0:
    print("Predicted State: Healthy Kidney")
else:
    print("Predicted State: Non-Viable Kidney")
print(f"Predicted States: {labels}")
total_duration = time.time() - start_time_pre
print(f"\nTotal Runtime: {total_duration:.2f} seconds")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DragDropArea()
    window.show()
    sys.exit(app.exec())