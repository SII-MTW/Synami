import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import shap
import torch.nn.functional as F

# ================= 🔧 Global Typography Configuration =================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42  # Ensure text exports as TrueType fonts, not paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PHYS_FILE = os.path.join(BASE_DIR, "data", "peptide_features_cleaned.csv")
TRAIN_ESM_FILE = os.path.join(BASE_DIR, "data", "peptide_esm_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "peptide_pure_vae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.pkl")

# Ensure results directory exists
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= 🧬 Feature Nomenclature Mapping =================
FEATURE_NAME_MAP = {
    'Hydrophobicity': 'Hydrophobicity',
    'Docking_Score': 'Docking Score',
    'dist_to_D147': 'Distance to D147 (Å)',
    'IFP_D147': 'IFP: D147',
    'IFP_S172': 'IFP: S172',
    'IFP_R277': 'IFP: R277',
    'IFP_T149': 'IFP: T149',
    'TPSA': 'TPSA (Å²)',
    'LogP': 'LogP',
    'pI': 'Isoelectric Point',
    'Net_Charge': 'Net Charge',
    'Length': 'Peptide Length',
    'MW': 'MW (Da)'
}

class PureVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(PureVAE, self).__init__()
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        h = self.encoder_base(x)
        return self.decoder(self.fc_mu(h)), self.fc_mu(h), self.fc_logvar(h)

def get_anchor_prototypes(model, scaler, device):
    df_phys, df_esm = pd.read_csv(TRAIN_PHYS_FILE), pd.read_csv(TRAIN_ESM_FILE)
    min_len = min(len(df_phys), len(df_esm))
    df_phys, df_esm = df_phys.iloc[:min_len], df_esm.iloc[:min_len]
    seqs = df_phys['Sequence'].values.tolist()
    indices = [seqs.index(s) for s in ['KGDEESLA', 'EDE'] if s in seqs]
    esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    esm_data = df_esm.iloc[indices][esm_cols].values.astype(np.float32)
    phys_data = scaler.transform(df_phys.iloc[indices][phys_cols].values.astype(np.float32))
    X_anchors = torch.tensor(np.hstack([esm_data, phys_data]), dtype=torch.float32).to(device)
    with torch.no_grad():
        _, mu_anchors, _ = model(X_anchors)
    return F.normalize(mu_anchors, p=2, dim=1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing High-Res SHAP Analysis Module on {device}...")
    
    scaler = joblib.load(SCALER_PATH)
    df_phys = pd.read_csv(TRAIN_PHYS_FILE)
    df_esm = pd.read_csv(TRAIN_ESM_FILE)
    min_len = min(len(df_phys), len(df_esm))
    df_phys, df_esm = df_phys.iloc[:min_len], df_esm.iloc[:min_len]
    
    esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    all_feature_names_raw = esm_cols + phys_cols
    input_dim = len(all_feature_names_raw)
    
    # Map technical feature names to academic nomenclature
    clean_feature_names = []
    for name in all_feature_names_raw:
        if name in FEATURE_NAME_MAP:
            clean_feature_names.append(FEATURE_NAME_MAP[name])
        elif name.startswith("ESM_"):
            dim_num = name.split('_')[1]
            clean_feature_names.append(f"ESM Embed (Dim {dim_num})")
        else:
            clean_feature_names.append(name.replace('_', ' '))
    
    esm_data_raw = df_esm[esm_cols].values.astype(np.float32)
    phys_data_raw = df_phys[phys_cols].values.astype(np.float32)
    X_train_raw = np.hstack([esm_data_raw, phys_data_raw])
    
    phys_data_scaled = scaler.transform(df_phys[phys_cols].values.astype(np.float32))
    X_train_scaled = np.hstack([esm_data_raw, phys_data_scaled])
    
    model = PureVAE(input_dim, 64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    anchors_norm = get_anchor_prototypes(model, scaler, device)
    
    class VAE_to_Score(nn.Module):
        def __init__(self, model, anchors_norm):
            super().__init__()
            self.model = model
            self.anchors_norm = anchors_norm
        def forward(self, x):
            _, mu, _ = self.model(x)
            Z_norm = F.normalize(mu, p=2, dim=1)
            sim_matrix = torch.mm(Z_norm, self.anchors_norm.transpose(0, 1))
            max_sim_scores, _ = torch.max(sim_matrix, dim=1)
            return (max_sim_scores + 1.0) / 2.0 * 40.0 

    wrapped_model = VAE_to_Score(model, anchors_norm).to(device)
    wrapped_model.eval()

    def predict_fn(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = wrapped_model(x_tensor)
        return scores.cpu().numpy()
    
    N_EXPLAIN_SAMPLES = min(1000, X_train_scaled.shape[0])
    N_BACKGROUND = 20 
    
    np.random.seed(42)
    
    print("[*] Extracting background centroids for SHAP estimation...")
    bg_samples_summary = shap.kmeans(X_train_scaled, N_BACKGROUND)
    
    sample_indices = np.random.choice(X_train_scaled.shape[0], N_EXPLAIN_SAMPLES, replace=False)
    test_samples_scaled = X_train_scaled[sample_indices]
    test_samples_raw = X_train_raw[sample_indices]
    
    print(f"[*] Calculating SHAP attribution values for {N_EXPLAIN_SAMPLES} samples (this may take 3-5 minutes)...")
    explainer = shap.KernelExplainer(predict_fn, bg_samples_summary)
    shap_values = explainer.shap_values(test_samples_scaled)
    if isinstance(shap_values, list): shap_values = shap_values[0]
        
    print("[*] Rendering high-resolution SHAP summary plot...")
    fig = plt.figure(figsize=(10, 8))
    
    # Custom color gradient (Oxford Blue -> Silver Gray -> Brick Red)
    colors = ["#1e466e", "#d9d9d9", "#b22222"] 
    cmap_nature = LinearSegmentedColormap.from_list("Nature_Style", colors)
    
    # Render SHAP Plot (limiting to Top 15 features for visual clarity)
    shap.summary_plot(
        shap_values, 
        features=test_samples_raw, 
        feature_names=clean_feature_names, 
        max_display=15, 
        show=False,
        cmap=cmap_nature,
        alpha=0.85
    )
    
    ax = plt.gca()
    
    # Minimalist Border Configuration
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    ax.tick_params(axis='both', which='major', width=1.5, length=6, labelsize=14)
    
    # Central reference line
    ax.axvline(x=0, color='#808080', linestyle='--', linewidth=1.5, zorder=0)
    
    ax.set_xlabel("SHAP Value (Impact on Synergistic Potential)", fontsize=16, fontweight='bold', labelpad=15)
    plt.title("Feature Importance Driving Umami AI Predictions", fontsize=20, fontweight='bold', pad=25, loc='left')
    
    plt.tight_layout()
    output_png = os.path.join(RESULTS_DIR, "Fig4_VAE_SHAP_Analysis.png")
    output_pdf = os.path.join(RESULTS_DIR, "Fig4_VAE_SHAP_Analysis.pdf")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print("[+] SHAP visualizations successfully generated.")
    print(f"    - Preview (PNG): {output_png}")
    print(f"    - Publication (PDF): {output_pdf}")

if __name__ == "__main__":
    main()