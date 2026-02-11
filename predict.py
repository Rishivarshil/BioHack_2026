import time
import torch
import pandas as pd
import joblib
import torch.nn as nn

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


# --- MANDATORY: Copy your class definitions here ---
# (KidneyAutoencoder and KidneyClassifier must be defined exactly as before)

# 1. Load the Scaler
scaler = joblib.load('data_scaler.pkl')

# 2. Re-initialize the architecture
# Note: We need the encoder to initialize the classifier
temp_ae = KidneyAutoencoder() 
model = KidneyClassifier(temp_ae.encoder)

# 3. Load the weights
model.load_state_dict(torch.load('kidney_classifier.pth'))
model.eval()

# 4. Predict on new data
def predict_new_data(csv_path):
    new_df = pd.read_csv(csv_path)
    
    # Preprocess exactly like training
    features = new_df.filter(regex='^dim_').values
    features_scaled = scaler.transform(features)
    input_tensor = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        # Get the 1.0/0.0 predictions
        raw_probs = model(input_tensor)
        predictions = (raw_probs > 0.5).int()
        
        # To get the reconstructed VECTOR (the SSL output)
        # We access the encoder inside the classifier
        latent_features = model.encoder(input_tensor)
    
    return predictions.flatten(), latent_features

# Usage
labels, vectors = predict_new_data('test_embeddings.csv')
print(f"Predicted States: {labels}")
total_duration = time.time() - start_time_pre
print(f"\nTotal Runtime: {total_duration:.2f} seconds")