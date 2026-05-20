import torch
import pandas as pd
import numpy as np
import os
import joblib
import torch.nn.functional as F

from model import PureVAE
from synergy_calculator import calculate_ortho_dynamic, calculate_allo_dynamic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEP_PHYS_FILE = os.path.join(BASE_DIR, "data", "test_features_final.csv")
PEP_ESM_FILE = os.path.join(BASE_DIR, "data", "test_esm_embeddings.csv")
CMP_PHYS_FILE = os.path.join(BASE_DIR, "data", "compounds_features_final.csv")
TRAIN_PHYS_FILE = os.path.join(BASE_DIR, "data", "peptide_features_cleaned.csv")
TRAIN_ESM_FILE = os.path.join(BASE_DIR, "data", "peptide_esm_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "peptide_pure_vae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_ORTHO = os.path.join(RESULTS_DIR, "test_synergy_ortho.csv")
OUTPUT_ALLO = os.path.join(RESULTS_DIR, "test_synergy_allo.csv")
LATENT_DIM = 64

def get_anchor_prototypes(model, scaler, device):
    """Extracts reference prototypes from the base training dataset to construct the scoring space."""
    df_phys = pd.read_csv(TRAIN_PHYS_FILE)
    df_esm = pd.read_csv(TRAIN_ESM_FILE)
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

def get_test_ai_scores(model, scaler, anchors_norm, device):
    """Calculates the AI base scores for the target test peptides."""
    df_phys = pd.read_csv(PEP_PHYS_FILE)
    df_esm = pd.read_csv(PEP_ESM_FILE)
    min_len = min(len(df_phys), len(df_esm))
    df_phys, df_esm = df_phys.iloc[:min_len], df_esm.iloc[:min_len]
    
    esm_cols = [c for c in df_esm.columns if c.startswith("ESM_")]
    phys_cols = [c for c in df_phys.columns if c not in ['ID', 'Sequence', 'Docking_File', 'Label', 'Fold']]
    
    esm_data = df_esm[esm_cols].values.astype(np.float32)
    phys_data = scaler.transform(df_phys[phys_cols].values.astype(np.float32)[:, :scaler.n_features_in_])
    
    X_test = torch.tensor(np.hstack([esm_data, phys_data]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _, mu_test, _ = model(X_test)
        Z_norm = F.normalize(mu_test, p=2, dim=1)
        sim_matrix = torch.mm(Z_norm, anchors_norm.transpose(0, 1))
        max_sim_scores, _ = torch.max(sim_matrix, dim=1)
        final_scores = (max_sim_scores.cpu().numpy() + 1.0) / 2.0 * 40.0 
        
    return df_phys, final_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing QSAR Synergy Evaluation Framework on {device}...")
    
    scaler = joblib.load(SCALER_PATH)
    df_temp_esm = pd.read_csv(PEP_ESM_FILE, nrows=1)
    input_dim = len([c for c in df_temp_esm.columns if c.startswith("ESM_")]) + scaler.n_features_in_
    
    model = PureVAE(input_dim, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    anchors_norm = get_anchor_prototypes(model, scaler, device)
    df_peps, ai_scores = get_test_ai_scores(model, scaler, anchors_norm, device)
    df_cmps = pd.read_csv(CMP_PHYS_FILE)
    
    results_ortho = []
    results_allo = []
    
    print(f"[*] Evaluating combinatorial synergy across compound library...")
    for i, row_pep in df_peps.iterrows():
        ai_val = ai_scores[i]
        for j, row_cmp in df_cmps.iterrows():
            site_type = row_cmp['Site_Type'] 
            
            res = {
                "Target_Peptide": row_pep['Sequence'],
                "Compound": row_cmp['Compounds'],
                "Mechanism": "Ortho" if site_type == 1 else "Allo",
                "u_Peptide": round(ai_val, 2)
            }
            
            if site_type == 1:
                score, detail = calculate_ortho_dynamic(ai_val, row_pep, row_cmp)
                res["Synergy_Index"] = round(score, 4)
                res["Detail"] = detail
                results_ortho.append(res)
            else:
                score, detail = calculate_allo_dynamic(ai_val, row_pep, row_cmp)
                res["Synergy_Index"] = round(score, 4)
                res["Detail"] = detail
                results_allo.append(res)
            
    if results_ortho:
        df_ortho = pd.DataFrame(results_ortho).sort_values(by=["Target_Peptide", "Synergy_Index"], ascending=[True, False])
        df_ortho.to_csv(OUTPUT_ORTHO, index=False)
        print(f"[+] Orthosteric synergy evaluation complete. Results saved to: {OUTPUT_ORTHO}")

    if results_allo:
        df_allo = pd.DataFrame(results_allo).sort_values(by=["Target_Peptide", "Synergy_Index"], ascending=[True, False])
        df_allo.to_csv(OUTPUT_ALLO, index=False)
        print(f"[+] Allosteric synergy evaluation complete. Results saved to: {OUTPUT_ALLO}")

if __name__ == "__main__":
    main()