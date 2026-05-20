# Deep Learning Framework for Peptide-Ligand Umami Synergy Prediction

This repository contains the source code for a predictive framework designed to evaluate the synergistic potential between umami peptides and small-molecule ligands. The pipeline integrates macroscopic physicochemical properties, structural interaction fingerprints (derived from molecular docking), and evolutionary sequence embeddings (ESM-2) into a unified Variational Autoencoder (VAE) and Quantitative Structure-Activity Relationship (QSAR) scoring architecture.

## Usage Guide

All execution scripts are located in the `src/` and `notebooks/` directories. Please ensure you run these scripts from the root directory of the repository to maintain relative path integrity.

### 1. Model Training
Train the Pure-VAE model to establish the latent manifold based on the extracted sequence and physicochemical features. This step generates the base AI scores for the peptides.
```bash
python src/train.py
```

### 2. Synergy Evaluation
Execute the core QSAR thermodynamic engine. This script calculates the synergistic potential (including both Orthosteric and Allosteric interactions) across the peptide and compound libraries, generating the final synergy indices.
```bash
python src/evaluate_synergy.py
```

### 3. Visualization and Analysis
Generate high-resolution, publication-ready plots to interpret the model's behavior and the underlying molecular mechanisms.
```bash
python notebooks/plot_tsne.py
python notebooks/plot_shap.py
python notebooks/plot_heatmap.py
python notebooks/plot_raincloud.py
```