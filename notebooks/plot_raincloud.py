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

FONT_TITLE  = 20
FONT_LABEL  = 16
FONT_TICK   = 14
FONT_TEXT   = 12

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
    """Extracts anchor prototypes from the base dataset to establish consistent scoring bounds."""
    df_phys, df_esm = pd.read_csv(TRAIN_PHYS_FILE), pd.read_csv(TRAIN_ESM_FILE)
    min_len = min(df_phys.shape[0], df_esm.shape[0])
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
    """Calculates AI scores for a given dataset."""
    min_len = min(df_phys.shape[0], df_esm.shape[0])
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

def get_abbr(seq):
    """Generates an abbreviation for long peptide sequences."""
    if len(seq) <= 4: return seq
    return f"{seq[0]}{seq[-1]}{len(seq)}"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing custom Raincloud visualization module on {device}...")
    
    scaler = joblib.load(SCALER_PATH)
    
    df_train_phys = pd.read_csv(TRAIN_PHYS_FILE)
    df_train_esm = pd.read_csv(TRAIN_ESM_FILE)
    df_test_phys = pd.read_csv(TEST_PHYS_FILE)
    df_test_esm = pd.read_csv(TEST_ESM_FILE)
    df_cmps = pd.read_csv(CMP_PHYS_FILE)

    esm_cols = [c for c in df_train_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_train_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    input_dim = len(esm_cols) + len(phys_cols)
    
    model = PureVAE(input_dim, 64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    anchors_norm = get_anchor_prototypes(model, scaler, device)
    
    print("[*] Calculating AI scores across datasets...")
    df_train_scored = get_ai_scores(df_train_phys, df_train_esm, model, scaler, anchors_norm, device)
    df_test_scored = get_ai_scores(df_test_phys, df_test_esm, model, scaler, anchors_norm, device)
    
    cmps_ortho = df_cmps[df_cmps['Site_Type'] == 1]
    cmps_allo = df_cmps[df_cmps['Site_Type'] == 2]

    print("[*] Computing background distribution for Raincloud plot...")
    bg_ortho_scores = []
    bg_allo_scores = []
    for _, pep_row in df_train_scored.iterrows():
        for _, cmp_row in cmps_ortho.iterrows():
            s, _ = calculate_ortho_dynamic(pep_row['AI_Score'], pep_row, cmp_row)
            bg_ortho_scores.append(s)
        for _, cmp_row in cmps_allo.iterrows():
            s, _ = calculate_allo_dynamic(pep_row['AI_Score'], pep_row, cmp_row)
            bg_allo_scores.append(s)

    target_names = ['glutamate', 'Succinate', 'IMP']
    df_target_cmps = df_cmps[df_cmps['Compounds'].isin(target_names)]
    
    highlight_data = []
    for _, pep_row in df_test_scored.iterrows():
        for _, cmp_row in df_target_cmps.iterrows():
            cmp_name = cmp_row['Compounds']
            mech = 'Orthosteric' if cmp_row['Site_Type'] == 1 else 'Allosteric'
            score, _ = calculate_ortho_dynamic(pep_row['AI_Score'], pep_row, cmp_row) if mech == 'Orthosteric' else calculate_allo_dynamic(pep_row['AI_Score'], pep_row, cmp_row)
            highlight_data.append({'Mech': mech, 'Score': score, 'Seq': pep_row['Sequence'], 'Cmp': cmp_name})
            
    df_highlight = pd.DataFrame(highlight_data)

    top_points = []
    for cmp_name in target_names:
        df_cmp = df_highlight[df_highlight['Cmp'] == cmp_name]
        if df_cmp.empty: continue
        max_record = df_cmp.loc[df_cmp['Score'].idxmax()]
        top_points.append(max_record)

    print("[*] Rendering Raincloud Plot...")
    fig, ax = plt.subplots(figsize=(7, 8)) 
    
    COLOR_ORTHO = '#9DB4C0' 
    COLOR_ALLO  = '#5C6B73'
    colors_viol = [COLOR_ORTHO, COLOR_ALLO]
    data_to_plot = [bg_ortho_scores, bg_allo_scores]
    positions = [0, 0.7]

    v_parts = ax.violinplot(data_to_plot, positions=positions,
                            showmeans=False, showextrema=False, widths=0.4)
    
    for pc, color in zip(v_parts['bodies'], colors_viol):
        paths = pc.get_paths()[0].vertices
        m = np.mean(paths[:, 0]) 
        paths[:, 0] = np.clip(paths[:, 0], m, np.inf)
        paths[:, 0] += 0.05
        
        pc.set_facecolor(color)
        pc.set_edgecolor('none')
        pc.set_alpha(0.6)

    b_parts = ax.boxplot(data_to_plot, positions=[p - 0.05 for p in positions], 
                         widths=0.06, showfliers=False, patch_artist=True)
    
    for patch in b_parts['boxes']:
        patch.set_facecolor('white') 
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    for median in b_parts['medians']:
        median.set(color='#d62728', linewidth=2)
    for cap, whisker in zip(b_parts['caps'], b_parts['whiskers']):
        cap.set(color='black', linewidth=1.5)
        whisker.set(color='black', linewidth=1.5)

    marker_colors = {'glutamate': '#d62728', 'Succinate': '#ff7f0e', 'IMP': '#2ca02c'}
    
    for p in top_points:
        abbr = get_abbr(p['Seq'])
        cmp_disp = "Glu" if p['Cmp'] == 'glutamate' else "Suc" if p['Cmp'] == 'Succinate' else "IMP"
        label_text = f"{abbr} + {cmp_disp}"
        color = marker_colors[p['Cmp']]
        
        x_base = positions[0] - 0.05 if p['Mech'] == 'Orthosteric' else positions[1] - 0.05 
        
        ax.scatter(x_base, p['Score'], color=color, marker='o', s=220, 
                   edgecolors='black', linewidth=1.5, zorder=20)
        
        if p['Mech'] == 'Orthosteric':
            x_text = x_base + 0.35 
            ha_align = 'left'
        else:
            x_text = x_base - 0.12 
            ha_align = 'right'
        
        ax.plot([x_base, x_text], [p['Score'], p['Score']], color='gray', linestyle=':', linewidth=1.5, zorder=5)
        
        ax.text(x_text, p['Score'], label_text, fontsize=FONT_TEXT+1, fontweight='bold', color='black',
                va='center', ha=ha_align, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2), zorder=21)

    ax.set_xlim(-0.3, 1.2) 
    ax.set_xticks(positions)
    ax.set_xticklabels(['Orthosteric', 'Allosteric'], fontweight='bold', fontsize=FONT_TICK)
    
    ax.set_xlabel("")
    ax.set_ylabel("Predicted Synergy Score", fontsize=FONT_LABEL, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_title("Synergy Breakthrough", fontweight='bold', fontsize=FONT_TITLE, pad=20)
    
    ax.grid(False) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False) 
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    output_png = os.path.join(RESULTS_DIR, "Fig3_Synergy_Raincloud_Perfect.png")
    output_pdf = os.path.join(RESULTS_DIR, "Fig3_Synergy_Raincloud_Perfect.pdf")
    
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf, dpi=300, format='pdf')
    
    print(f"[+] Raincloud visualization successfully generated.")
    print(f"    - Preview (PNG): {output_png}")
    print(f"    - Publication (PDF): {output_pdf}")

if __name__ == "__main__":
    main()