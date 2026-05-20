import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
import torch.nn.functional as F

# ================= 🔧 Global Configuration =================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PHYS_FILE = os.path.join(BASE_DIR, "data", "peptide_features_cleaned.csv")
TRAIN_ESM_FILE = os.path.join(BASE_DIR, "data", "peptide_esm_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "peptide_pure_vae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.pkl")

# Ensure results directory exists
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ==========================================================

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

def get_quadrant(x, y, x_mean, y_mean):
    if x > x_mean and y > y_mean: return "Top-Right"
    if x < x_mean and y > y_mean: return "Top-Left"
    if x < x_mean and y < y_mean: return "Bottom-Left"
    if x > x_mean and y < y_mean: return "Bottom-Right"
    return "Center"

def distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing VAE t-SNE Analysis Module on {device}...")
    
    scaler = joblib.load(SCALER_PATH)
    df_phys = pd.read_csv(TRAIN_PHYS_FILE)
    df_esm = pd.read_csv(TRAIN_ESM_FILE)
    min_len = min(len(df_phys), len(df_esm))
    df_phys, df_esm = df_phys.iloc[:min_len], df_esm.iloc[:min_len]
    
    esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    input_dim = len(esm_cols) + len(phys_cols)
    
    model = PureVAE(input_dim, 64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    anchors_norm = get_anchor_prototypes(model, scaler, device)
    
    esm_data = df_esm[esm_cols].values.astype(np.float32)
    phys_data = scaler.transform(df_phys[phys_cols].values.astype(np.float32))
    X_train = torch.tensor(np.hstack([esm_data, phys_data]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, mu_train, _ = model(X_train)
        Z_norm = F.normalize(mu_train, p=2, dim=1)
        sim_matrix = torch.mm(Z_norm, anchors_norm.transpose(0, 1))
        max_sim_scores, _ = torch.max(sim_matrix, dim=1)
        ai_scores = (max_sim_scores.cpu().numpy() + 1.0) / 2.0 * 40.0
        
    df_phys['AI_Score'] = ai_scores
    mu_train_np = mu_train.cpu().numpy()
    
    print("[*] Computing t-SNE dimensionality reduction (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(mu_train_np)
    
    # Calculate chart boundaries for collision detection
    x_range = z_tsne[:, 0].max() - z_tsne[:, 0].min()
    y_range = z_tsne[:, 1].max() - z_tsne[:, 1].min()
    min_safe_distance = max(x_range, y_range) * 0.12  # Adjust buffer to maintain score gradient clarity
    
    x_mean, y_mean = np.mean(z_tsne[:, 0]), np.mean(z_tsne[:, 1])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Render scatter plot background
    scatter = ax.scatter(z_tsne[:, 0], z_tsne[:, 1], c=df_phys['AI_Score'], 
                         cmap='viridis', alpha=0.85, edgecolors='none', s=80)
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Predicted AI Umami Potential (u_Peptide)', fontsize=24, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=18)
    
    # ================= 🧬 Score-Based Equidistant Sampling Logic =================
    selected_points = [] 
    target_seqs = []     
    
    # 1. Force inclusion of anchor peptides
    for anchor in ['KGDEESLA', 'EDE']:
        idx_match = df_phys[df_phys['Sequence'] == anchor].index
        if len(idx_match) > 0:
            idx = idx_match[0]
            x, y = z_tsne[idx, 0], z_tsne[idx, 1]
            selected_points.append((x, y))
            target_seqs.append(anchor)

    df_pool = df_phys[~df_phys['Sequence'].isin(target_seqs)].sort_values('AI_Score', ascending=False)
    
    max_score = df_pool['AI_Score'].max()
    min_score = df_pool['AI_Score'].min()
    
    # Target score brackets for uniform sampling
    target_scores = [
        max_score - (max_score - min_score) * 0.25,
        max_score - (max_score - min_score) * 0.50,
        max_score - (max_score - min_score) * 0.75,
        min_score
    ]

    for target_score in target_scores:
        df_pool['score_diff'] = abs(df_pool['AI_Score'] - target_score)
        df_sorted_by_diff = df_pool.sort_values('score_diff')
        
        found = False
        for _, row in df_sorted_by_diff.iterrows():
            seq = row['Sequence']
            if seq in target_seqs:
                continue
                
            idx = df_phys[df_phys['Sequence'] == seq].index[0]
            curr_p = (z_tsne[idx, 0], z_tsne[idx, 1])
            
            # Distance Validation to prevent overlap
            is_safe = True
            for sp in selected_points:
                if distance(curr_p, sp) < min_safe_distance:
                    is_safe = False
                    break
            
            if is_safe:
                selected_points.append(curr_p)
                target_seqs.append(seq)
                found = True
                break 
        
        # Relax distance constraints slightly if exact separation is impossible
        if not found and not df_sorted_by_diff.empty:
            for _, row in df_sorted_by_diff.iterrows():
                 seq = row['Sequence']
                 if seq in target_seqs: continue
                 idx = df_phys[df_phys['Sequence'] == seq].index[0]
                 curr_p = (z_tsne[idx, 0], z_tsne[idx, 1])
                 
                 is_safe = True
                 for sp in selected_points:
                    if distance(curr_p, sp) < min_safe_distance * 0.5:
                        is_safe = False
                        break
                 if is_safe:
                     selected_points.append(curr_p)
                     target_seqs.append(seq)
                     break
                
    # Gather plotting annotations
    plot_info = []
    for seq in target_seqs:
        idx = df_phys[df_phys['Sequence'] == seq].index[0]
        x, y = z_tsne[idx, 0], z_tsne[idx, 1]
        score = df_phys.iloc[idx]['AI_Score']
        quad = get_quadrant(x, y, x_mean, y_mean)
        plot_info.append({'Sequence': seq, 'Score': score, 'X': x, 'Y': y, 'Quadrant': quad})
    
    plot_info = sorted(plot_info, key=lambda d: d['Score'], reverse=True)
    
    print("\n" + "="*70)
    print("[*] Presentation Guide: Locate points with bold black outlines.")
    print("[*] Equidistant sampling enabled. Score gradients are isolated.")
    print("")
    
    for info in plot_info:
        color_rgba = scatter.cmap(scatter.norm(info['Score']))
        ax.scatter(info['X'], info['Y'], color=color_rgba, s=100, edgecolors='black', linewidth=2.5, zorder=10)
        
        is_anchor = " (Anchor)" if info['Sequence'] in ['KGDEESLA', 'EDE'] else ""
        print(f"  - Seq: {info['Sequence']:<12} {is_anchor:<10} | AI Score: {info['Score']:4.1f} | Quadrant: {info['Quadrant']}")
        
    print("="*70 + "\n")

    # ================= 📏 Typography & High-Res Formatting =================
    ax.set_title("Latent Space Segregation of Peptide Umami Potential", fontsize=28, fontweight='bold', pad=25)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=24, fontweight='bold', labelpad=15)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=24, fontweight='bold', labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    
    output_img = os.path.join(RESULTS_DIR, "Fig1_VAE_tSNE_Clustering.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[+] t-SNE clustering visualization successfully generated: {output_img}")

if __name__ == "__main__":
    main()