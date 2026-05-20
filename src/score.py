import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os
import joblib
from model import PureVAE
from train import PureVAEDataset, FILE_PHYS, FILE_ESM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "peptide_pure_vae.pth")
SCALER_PATH = os.path.join(BASE_DIR, "checkpoints", "feature_scaler.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_RANKING = os.path.join(RESULTS_DIR, "final_ranking.csv")

LATENT_DIM = 64
BATCH_SIZE = 32
TARGET_ANCHORS = ['KGDEESLA', 'EDE'] 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Calculating Multi-Prototype Max Cosine Similarity on {device}")

    dataset = PureVAEDataset(FILE_PHYS, FILE_ESM)
    input_dim = dataset.X.shape[1]
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PureVAE(input_dim, LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_zs = []
    
    with torch.no_grad():
        for batch_x in full_loader:
            batch_x = batch_x.to(device)
            _, mu, _ = model(batch_x)
            all_zs.append(mu.cpu())
            
    Z_matrix = torch.cat(all_zs, dim=0) 
    all_seqs_list = list(dataset.seqs)

    found_indices = []
    found_seqs = []
    
    for seq in TARGET_ANCHORS:
        if seq in all_seqs_list:
            found_indices.append(all_seqs_list.index(seq))
            found_seqs.append(seq)

    if not found_indices:
        raise ValueError("Critical Error: Anchor peptides not found in dataset.")
    
    print(f"[*] Established prototypes: {found_seqs}")

    Z_anchors = Z_matrix[found_indices] 

    Z_norm = F.normalize(Z_matrix, p=2, dim=1)
    A_norm = F.normalize(Z_anchors, p=2, dim=1)
    
    sim_matrix = torch.mm(Z_norm, A_norm.transpose(0, 1))
    
    max_sim_scores, best_anchor_idx = torch.max(sim_matrix, dim=1)
    
    normalized_scores = (max_sim_scores.numpy() + 1.0) / 2.0 * 40.0 

    closest_anchor_names = [found_seqs[idx] for idx in best_anchor_idx.numpy()]

    df_rank = pd.DataFrame({
        'Peptide': dataset.seqs, 
        'Score': normalized_scores,
        'Closest_Type': closest_anchor_names
    })
    
    df_rank = df_rank.sort_values('Score', ascending=False)
    df_rank.to_csv(OUTPUT_RANKING, index=False)
    
    print(f"[+] Ranking complete. Results saved to: {OUTPUT_RANKING}")
    
    print("\n[*] Top 5 Umami Potential Peptides:")
    print(df_rank[['Peptide', 'Score', 'Closest_Type']].head(5).to_string(index=False))
    
    print("\n[*] Prototype Ranking Validation:")
    for seq in found_seqs:
        anchor_data = df_rank[df_rank['Peptide'] == seq]
        rank = df_rank.index.get_loc(anchor_data.index[0]) + 1
        score = anchor_data.iloc[0]['Score']
        print(f"  - {seq} Rank: #{rank} (Score: {score:.2f})")

if __name__ == "__main__":
    main()