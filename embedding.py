import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models
from tqdm import tqdm

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
    # 1. Load the image as a NumPy array
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    # 2. Convert to grayscale and threshold to find the 'data area'
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # 3. Find contours and only proceed if they exist
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour (the ultrasound cone)
        c = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the organ area
        x, y, w, h = cv2.boundingRect(c)
        
        # 4. PHYSICAL CROP: Remove the black void entirely
        cropped_img = img_bgr[y:y+h, x:x+w]
        
        # 5. MASK: Zero out any remaining black corners within the crop
        mask = np.zeros((h, w), dtype=np.uint8)
        # Shift the contour coordinates to match the new crop
        c_shifted = c - [x, y]
        cv2.drawContours(mask, [c_shifted], -1, 255, -1)
        final_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)
    else:
        final_img = img_bgr

    # Convert to RGB for the PyTorch FeatureExtractor
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
base_data_dir = "./test"
modalities = ["ultrasound", "oct"]
states = {"kidney": 0, "tumor": 1}  
all_data = []

model = FeatureExtractor()
model.eval() 
for mod in modalities:
    for state_name, label in states.items():
        folder_path = os.path.join(base_data_dir, mod, state_name)
        
        # Ensure path exists before looping
        if not os.path.exists(folder_path):
            print(f"Skipping missing directory: {folder_path}")
            continue
            
        for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {mod}/{state_name}"):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    if mod == "ultrasound":
                        img = strict_medical_crop(img_path)
                    else:
                        img = Image.open(img_path).convert('RGB')
                    
                    if img is None: continue

                    input_tensor = preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        vector = model(input_tensor).squeeze().cpu().numpy()
                    
                    entry = {
                        "modality": mod,
                        "state": state_name,
                        "label": label,
                        "path": img_path
                    }
                    for i, val in enumerate(vector):
                        entry[f"dim_{i}"] = val
                    
                    all_data.append(entry)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

df = pd.DataFrame(all_data)
df.to_csv("test_embeddings.csv", index=False)
print(f"Saved {len(df)} embeddings to test_embeddings.csv")


