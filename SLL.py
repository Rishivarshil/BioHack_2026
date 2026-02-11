import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Load and Preprocess Data
df = pd.read_csv('organ_embeddings.csv')  

# Drop non-numeric and target columns for SSL
features = df.filter(regex='^dim_').values 

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train = torch.FloatTensor(features_scaled)
dataset = TensorDataset(X_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

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

model = KidneyAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    for batch in loader:
        inputs = batch[0]
        
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Self-supervised pre-training complete.")

y_labels = torch.FloatTensor(df['label'].values).view(-1, 1)
labeled_dataset = TensorDataset(X_train, y_labels)
labeled_loader = DataLoader(labeled_dataset, batch_size=8, shuffle=True)

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


classifier = KidneyClassifier(model.encoder)
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.0005)

epochs = 50
for epoch in range(epochs):
    for inputs, labels in labeled_loader:
        predictions = classifier(inputs)
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Fine-tuning Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Classifier is ready.")
torch.save(classifier.state_dict(), 'kidney_classifier.pth')
import joblib
joblib.dump(scaler, 'data_scaler.pkl')

print("Model and Scaler saved successfully.")
