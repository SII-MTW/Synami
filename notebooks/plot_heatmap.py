import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import torch.nn.functional as F

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['pdf.fonttype'] = 42 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

TRAIN_PHYS_FILE = os.path.join(BASE_DIR, "data", "peptide_features_cleaned.csv")
TRAIN_ESM_FILE = os.path.join(BASE_DIR, "data", "peptide_esm_cleaned.csv")

TEST_PHYS_FILE = os.path.join(BASE_DIR, "data", "test_features_final.csv")
TEST_ESM_FILE = os.path.join(BASE_DIR, "data", "test_esm_embeddings.csv")

CMP_PHYS_FILE = os.path.join(BASE_DIR, "data", "compounds_features_final.csv")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "peptide_pure_vae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.pkl")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from model import PureVAE
from synergy_calculator import calculate_ortho_dynamic, calculate_allo_dynamic

def get_anchor_prototypes(model, scaler, device):
    """Extract anchors from the base library to ensure consistent scoring space."""
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

def get_ai_scores(df_phys, df_esm, model, scaler, anchors_norm, device):
    """Utility function to calculate AI scores for an input dataset."""
    min_len = min(len(df_phys), len(df_esm))
    df_phys, df_esm = df_phys.iloc[:min_len].copy(), df_esm.iloc[:min_len].copy()
    
    esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold', 'AI_Score']]
    
    esm_data = df_esm[esm_cols].values.astype(np.float32)
    phys_data = scaler.transform(df_phys[phys_cols].values.astype(np.float32))
    
    X_tensor = torch.tensor(np.hstack([esm_data, phys_data]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, mu, _ = model(X_tensor)
        Z_norm = F.normalize(mu, p=2, dim=1)
        sim_matrix = torch.mm(Z_norm, anchors_norm.transpose(0, 1))
        scores = (torch.max(sim_matrix, dim=1)[0].cpu().numpy() + 1.0) / 2.0 * 40.0
    
    df_phys['AI_Score'] = scores
    return df_phys

def build_synergy_matrices(df_peps, df_cmps):
    """Utility function to build Orthosteric and Allosteric matrices, ordered by column means."""
    cmps_ortho = df_cmps[df_cmps['Site_Type'] == 1]
    cmps_allo = df_cmps[df_cmps['Site_Type'] == 2]
    
    def build_matrix(df_p, df_c, mode='ortho'):
        data = []
        for _, pep_row in df_p.iterrows():
            row_dict = {'Peptide': pep_row['Sequence']}
            for _, cmp_row in df_c.iterrows():
                if mode == 'ortho':
                    score, _ = calculate_ortho_dynamic(pep_row['AI_Score'], pep_row, cmp_row)
                else:
                    score, _ = calculate_allo_dynamic(pep_row['AI_Score'], pep_row, cmp_row)
                row_dict[cmp_row['Compounds']] = score
            data.append(row_dict)
        return pd.DataFrame(data).set_index('Peptide')

    df_ortho = build_matrix(df_peps, cmps_ortho, 'ortho')
    df_allo = build_matrix(df_peps, cmps_allo, 'allo')
    
    df_ortho = df_ortho[df_ortho.mean().sort_values(ascending=False).index]
    df_allo = df_allo[df_allo.mean().sort_values(ascending=False).index]
    
    return df_ortho, df_allo

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Macro/Micro Synergy Heatmap rendering module on {device}...")
    
    scaler = joblib.load(SCALER_PATH)
    
    df_train_phys = pd.read_csv(TRAIN_PHYS_FILE)
    df_train_esm = pd.read_csv(TRAIN_ESM_FILE)
    df_test_phys = pd.read_csv(TEST_PHYS_FILE)
    df_test_esm = pd.read_csv(TEST_ESM_FILE)
    df_cmps = pd.read_csv(CMP_PHYS_FILE)
    
    if 'Site_Type' not in df_cmps.columns:
        print("[!] Warning: 'Site_Type' column not found in compounds file.")
        return 

    esm_cols = [c for c in df_train_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_train_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    input_dim = len(esm_cols) + len(phys_cols)
    
    model = PureVAE(input_dim, 64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    anchors_norm = get_anchor_prototypes(model, scaler, device)
    
    print("[*] Calculating synergy scores for the Base Library...")
    df_train_scored = get_ai_scores(df_train_phys, df_train_esm, model, scaler, anchors_norm, device)
    df_train_top = df_train_scored.sort_values('AI_Score', ascending=False).head(30)
    train_ortho, train_allo = build_synergy_matrices(df_train_top, df_cmps)
    
    print("[*] Calculating synergy scores for Target Core Peptides...")
    df_test_scored = get_ai_scores(df_test_phys, df_test_esm, model, scaler, anchors_norm, device)
    df_test_top = df_test_scored.sort_values('AI_Score', ascending=False)
    test_ortho, test_allo = build_synergy_matrices(df_test_top, df_cmps)

    print("[*] Rendering 2x2 Panorama Heatmap...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), gridspec_kw={'wspace': 0.05, 'hspace': 0.25, 'height_ratios': [4, 1]})
    cmap_choice = "Blues" 
    
    sns.heatmap(train_ortho, cmap=cmap_choice, ax=axes[0, 0], 
                cbar_kws={'label': 'Orthosteric Score', 'shrink': 0.8})
    axes[0, 0].set_title("Orthosteric Synergy", fontsize=20, fontweight='bold', pad=15)
    axes[0, 0].set_ylabel("Target Peptides", fontsize=16, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].collections[0].colorbar.set_label('Orthosteric Score', fontsize=14, fontweight='bold')

    sns.heatmap(train_allo, cmap=cmap_choice, ax=axes[0, 1], 
                cbar_kws={'label': 'Allosteric Score', 'shrink': 0.8})
    axes[0, 1].set_title("Allosteric Synergy", fontsize=20, fontweight='bold', pad=15)
    axes[0, 1].set_ylabel("")
    axes[0, 1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0, 1].tick_params(axis='y', left=False, labelleft=False) 
    axes[0, 1].collections[0].colorbar.set_label('Allosteric Score', fontsize=14, fontweight='bold')

    sns.heatmap(test_ortho, cmap=cmap_choice, ax=axes[1, 0], 
                cbar_kws={'label': 'Orthosteric Score', 'shrink': 0.8},
                annot=True, fmt=".1f", annot_kws={"size": 10}, 
                linewidths=0.5, linecolor='lightgray') 
    
    axes[1, 0].set_ylabel("Predicted Peptides", fontsize=16, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1, 0].tick_params(axis='y', labelsize=14, rotation=0)
    axes[1, 0].collections[0].colorbar.set_label('Orthosteric Score', fontsize=14, fontweight='bold')

    sns.heatmap(test_allo, cmap=cmap_choice, ax=axes[1, 1], 
                cbar_kws={'label': 'Allosteric Score', 'shrink': 0.8},
                annot=True, fmt=".1f", annot_kws={"size": 10},
                linewidths=0.5, linecolor='lightgray')
    
    axes[1, 1].set_ylabel("")
    axes[1, 1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1, 1].tick_params(axis='y', left=False, labelleft=False)
    axes[1, 1].collections[0].colorbar.set_label('Allosteric Score', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_png = os.path.join(RESULTS_DIR, "Fig2_Macro_Micro_Synergy_Heatmap.png")
    output_pdf = os.path.join(RESULTS_DIR, "Fig2_Macro_Micro_Synergy_Heatmap.pdf")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"[+] Macro/Micro Synergy Heatmap successfully generated.")
    print(f"    - Preview (PNG): {output_png}")
    print(f"    - Publication (PDF): {output_pdf}")

if __name__ == "__main__":
    main()