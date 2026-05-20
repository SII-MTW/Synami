import pandas as pd

QSAR_CONFIG = {
    # ---------------- 1. Orthosteric Site Physicochemical Mechanics ----------------
    "ORTHO_POLAR_DENSITY_WEIGHT": 10.0,   
    "ORTHO_CHARGE_BRIDGE_WEIGHT": 0.5,    
    "ORTHO_CAPACITY_BASE": 4.0,           
    "ORTHO_MW_NORMALIZATION": 110.0,      
    "ORTHO_STERIC_PENALTY_WEIGHT": 1.5,   
    "ORTHO_PREORG_SCALE": 10.0, 
    
    # ---------------- 2. Allosteric Site Penalties & Specificity ----------------
    "WEIGHT_ENTROPY": 0.5,   
    "WEIGHT_DESOLVATION": 0.04,   
    "WEIGHT_LOGP_PENALTY": 2.0,
    "WEIGHT_IMBALANCE": 3.0,               

    # ---------------- 3. Mathematical Scaling ----------------
    "LE_SCALE_FACTOR": 1000.0,      
    
    # ---------------- 4. Peptide Base Thermodynamics ----------------
    "WEIGHT_IFP": 1.5, 
    "PEP_BASE_WEIGHT": 0.1,                        
}

ORTHO_KEY_RESIDUES = ['IFP_ASP147', 'IFP_THR149', 'IFP_SER172', 'IFP_GLU301']
ALLO_KEY_RESIDUES = ['IFP_HIS71', 'IFP_SER306', 'IFP_ARG277']

def calculate_ortho_dynamic(pep_ai_score, pep_info, cmp_info):
    """
    Calculates the dynamic synergy score for orthosteric interactions based on
    physicochemical features and peptide-compound competition dynamics.
    """
    docking = abs(cmp_info.get('Docking_Score', 0.0))
    mw = max(10.0, cmp_info.get('MW', 100.0))
    logp = cmp_info.get('LogP', 0.0)
    tpsa = cmp_info.get('TPSA', 100.0)
    h_donors = cmp_info.get('H_Donors', 0.0)
    
    le_score = (docking / mw) * QSAR_CONFIG["LE_SCALE_FACTOR"]
    lle_score = docking - logp
    polar_density = (tpsa / mw) * QSAR_CONFIG["ORTHO_POLAR_DENSITY_WEIGHT"]
    hit_count = sum(1 for res in ORTHO_KEY_RESIDUES if cmp_info.get(res, 0) == 1)
    
    pep_dock = abs(pep_info.get('Docking_Score', 0.0))
    pep_charge = pep_info.get('Net_Charge', 0.0)
    pep_len = pep_info.get('Length', 7)
    
    charge_match_bonus = abs(pep_charge) * h_donors * QSAR_CONFIG["ORTHO_CHARGE_BRIDGE_WEIGHT"]
    
    steric_restriction_penalty = max(0, pep_len - QSAR_CONFIG["ORTHO_CAPACITY_BASE"]) * \
                                 (mw / QSAR_CONFIG["ORTHO_MW_NORMALIZATION"]) * \
                                 QSAR_CONFIG["ORTHO_STERIC_PENALTY_WEIGHT"]
    
    ifp_multiplier = 1.0 + (pep_dock / QSAR_CONFIG["ORTHO_PREORG_SCALE"])
    ifp_synergy = hit_count * QSAR_CONFIG["WEIGHT_IFP"] * ifp_multiplier
    
    v_cmp = le_score + lle_score + polar_density + ifp_synergy + charge_match_bonus - steric_restriction_penalty
    
    pep_bonus = pep_ai_score * QSAR_CONFIG["PEP_BASE_WEIGHT"]
    
    total = pep_bonus + v_cmp
    detail = f"LE={le_score:.1f}, PolarDens={polar_density:.1f}, ChargeMatch={charge_match_bonus:.1f}, StericPen={-steric_restriction_penalty:.1f}"
    
    return total, detail

def calculate_allo_dynamic(pep_ai_score, pep_info, cmp_info):
    """
    Calculates the dynamic synergy score for allosteric interactions driven by
    continuous thermodynamic features and conformational entropy penalties.
    """
    docking = abs(cmp_info.get('Docking_Score', 0.0))
    mw = max(10.0, cmp_info.get('MW', 100.0))
    logp = cmp_info.get('LogP', 0.0)
    tpsa = cmp_info.get('TPSA', 100.0)
    h_acc = cmp_info.get('H_Acceptors', 0.0)
    h_don = cmp_info.get('H_Donors', 0.0)
    rot_bonds = cmp_info.get('Rotatable_Bonds', 4.0)
    
    le_score = (docking / mw) * QSAR_CONFIG["LE_SCALE_FACTOR"]
    
    imbalance = max(0, h_acc - h_don)
    imbal_penalty = imbalance * QSAR_CONFIG["WEIGHT_IMBALANCE"]
    
    logp_penalty = logp * QSAR_CONFIG["WEIGHT_LOGP_PENALTY"]
    tpsa_penalty = tpsa * QSAR_CONFIG["WEIGHT_DESOLVATION"]
    rot_penalty = rot_bonds * QSAR_CONFIG["WEIGHT_ENTROPY"]
    
    hit_count = sum(1 for res in ALLO_KEY_RESIDUES if cmp_info.get(res, 0) == 1)
    ifp_bonus = hit_count * QSAR_CONFIG["WEIGHT_IFP"]
    
    v_cmp = le_score + ifp_bonus + logp_penalty - imbal_penalty - tpsa_penalty - rot_penalty
    pep_bonus = pep_ai_score * QSAR_CONFIG["PEP_BASE_WEIGHT"]
    
    dynamic_gamma = 1.0 + (hit_count / len(ALLO_KEY_RESIDUES))
    synergy_burst = dynamic_gamma * v_cmp 
    
    total = pep_bonus + v_cmp + synergy_burst 
    
    detail = f"LE={le_score:.1f}, Imbal={imbalance}, TPSA_Pen={-tpsa_penalty:.1f}, Burst={synergy_burst:.1f}"
    
    return total, detail