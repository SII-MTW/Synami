import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from model import PureVAE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PHYS = os.path.join(BASE_DIR, "data", "peptide_features_cleaned.csv")
FILE_ESM = os.path.join(BASE_DIR, "data", "peptide_esm_cleaned.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "peptide_pure_vae.pth")
SCALER_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "feature_scaler.pkl")

LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 200        
LEARNING_RATE = 1e-3
W_RECON = 1.0       
W_KL = 0.001        

class PureVAEDataset(Dataset):
    def __init__(self, phys_file, esm_file):
        try:
            df_phys = pd.read_csv(phys_file)
            df_esm = pd.read_csv(esm_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing data files:\n{phys_file}\n{esm_file}")
        
        min_len = min(len(df_phys), len(df_esm))
        df_phys = df_phys.iloc[:min_len]
        df_esm = df_esm.iloc[:min_len]

        esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
        exclude_cols = ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']
        phys_cols = [c for c in df_phys.columns if c not in exclude_cols]
        
        self.esm_data = df_esm[esm_cols].values.astype(np.float32)
        self.phys_data = df_phys[phys_cols].values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.phys_data = self.scaler.fit_transform(self.phys_data)
        self.X = np.hstack([self.esm_data, self.phys_data])
        
        self.seqs = df_phys['Sequence'].values

    def __len__(self): 
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Pure VAE for Latent Representation on {device}")

    dataset = PureVAEDataset(FILE_PHYS, FILE_ESM)
    joblib.dump(dataset.scaler, SCALER_SAVE_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    input_dim = dataset.X.shape[1]

    model = PureVAE(input_dim, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_recon = 0
        total_kl = 0
        
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            
            recon, mu, logvar = model(batch_x)
            
            loss_recon = nn.functional.mse_loss(recon, batch_x)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
            
            loss = W_RECON * loss_recon + W_KL * loss_kl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_recon += loss_recon.item()
            total_kl += loss_kl.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Recon Loss: {total_recon/len(dataloader):.4f} | KL Loss: {total_kl/len(dataloader):.4f}")

    print(f"[*] Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("[+] VAE weights saved successfully.")

if __name__ == "__main__":
    main()