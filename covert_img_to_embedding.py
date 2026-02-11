import sys
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models

if len(sys.argv) > 1:
    # This captures the 'path' sent by subprocess.run
    image_path = sys.argv[1]
    print(f"GUI provided image path: {image_path}")
else:
    # Fallback if you run the script manually without the GUI
    image_path = input("Enter the path to your image: ")
    
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

