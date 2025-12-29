"""
Docstring for vae.cmp_gen_baseline2chess
Compares CHESS simulation (Real) vs VAE (MLP) generation for Low, Mid, High mutation ranges.
Reads parameters and ground truth from CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy import linalg

# Ensure we can import model from the same directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from model import TumorEvolutionModel

# Configuration
PROJECT_ROOT = Path("/root/wja/wja/project/CHESS.cpp")
VAE_CHECKPOINT_PATH = Path('/root/data/wja/project/CHESS.cpp/vae/ckpts/vae_checkpoint.pth')
CSV_PATH = Path("/root/wja/wja/project/CHESS.cpp/vae/vae_out/vaf_comparison_data.csv")
OUTPUT_DIR = Path("/root/wja/wja/project/CHESS.cpp/vae/vae_out/cmp_new")

def get_vae_vaf_distribution(row, device, model):
    # Prepare input
    # Order: mutation_rates, birth_rates, death_rates, aggressions, time, father, universe
    
    inputs = []
    
    # Clone 1
    c1_params = [
        float(row['C1_Mutation_Rate']),
        float(row['C1_Birth_Rate']),
        0.0, # death_rates
        float(row['C1_Aggression']),
        float(row['C1_Start_Time']),
        0.0, # father
        0.0  # universe
    ]
    inputs.append(c1_params)
    
    # Clone 2 (if Double)
    if row['Clone_Type'] == 'Double':
        c2_params = [
            float(row['C2_Mutation_Rate']),
            float(row['C2_Birth_Rate']),
            0.0,
            float(row['C2_Aggression']),
            float(row['C2_Start_Time']),
            0.0,
            0.0
        ]
        inputs.append(c2_params)
    
    x_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out = model(x_tensor)
        mu = out[..., 0]
        sigma = out[..., 1]
        sampled = torch.normal(mu, sigma).cpu().numpy()
        
    return sampled

def calculate_fid(act1, act2):
    """
    Calculate Frechet Inception Distance between two distributions.
    Here we use the VAF vectors directly as features.
    act1, act2: numpy arrays of shape (N, 100)
    """
    # Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2)
    
    # Calculate sqrt of product of covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main():
    if not CSV_PATH.exists():
        print(f"CSV file not found at {CSV_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Data
    print(f"Reading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Filter for Real Source only (we will generate the comparison)
    df_filtered = df[df['Source'] == 'Real (CHESS)']
    
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not VAE_CHECKPOINT_PATH.exists():
        print(f"VAE checkpoint not found at {VAE_CHECKPOINT_PATH}")
        return
        
    print(f"Loading model from {VAE_CHECKPOINT_PATH}")
    model = TumorEvolutionModel(input_dim=7).to(device)
    try:
        checkpoint = torch.load(VAE_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()
    
    predicted_data = []
    
    # Lists to collect vectors for FID calculation
    real_vectors = []
    meanflow_vectors = []
    pred_vectors = []

    print("Processing samples...")
    for idx, row in df_filtered.iterrows():
        sample_id = row['Sample_ID']
        clone_type = row['Clone_Type']
        mutation_range = row['Mutation_Range']
        
        print(f"  - Processing {clone_type} - {mutation_range} (Sample {sample_id})")
        
        # Extract Real VAF
        bin_cols = [f'Bin_{i}' for i in range(100)]
        real_vaf = row[bin_cols].to_numpy(dtype=float)

        # Extract MeanFlow VAF
        meanflow_row = df[(df['Sample_ID'] == sample_id) & (df['Source'] == 'Generated (MeanFlow)')]
        meanflow_vaf = None
        if not meanflow_row.empty:
            meanflow_vaf = meanflow_row.iloc[0][bin_cols].to_numpy(dtype=float)
        
        # Run VAE
        pred_vaf = get_vae_vaf_distribution(row, device, model)
        
        # Collect vectors
        real_vectors.append(real_vaf)
        pred_vectors.append(pred_vaf)
        if meanflow_vaf is not None:
            meanflow_vectors.append(meanflow_vaf)
        
        # Store predicted data
        pred_record = {
            'Sample_ID': sample_id,
            'Clone_Type': clone_type,
            'Mutation_Range': mutation_range,
            **{f'Bin_{i}': pred_vaf[i] for i in range(100)}
        }
        predicted_data.append(pred_record)
        
        # Plot
        plt.figure(figsize=(10, 6))
        x_axis = np.arange(100)
        
        plt.plot(x_axis, real_vaf, color='tab:blue', label='CHESS (Simulation)', linewidth=2)
        if meanflow_vaf is not None:
            plt.plot(x_axis, meanflow_vaf, color='tab:green', label='MeanFlow', linewidth=2, linestyle='-.')
        plt.plot(x_axis, pred_vaf, color='tab:orange', label='MLP-VAE', linewidth=2, linestyle='--')
        
        title = (f"[{clone_type} (MLP) - {mutation_range}] Sample {sample_id}\n"
                 f"C1: Mut={row['C1_Mutation_Rate']:.2f}, Birth={row['C1_Birth_Rate']:.2f}, "
                 f"Agg={row['C1_Aggression']:.2f}, T={row['C1_Start_Time']:.2f}")
        
        if clone_type == 'Double':
            title += (f"\nC2: Mut={row['C2_Mutation_Rate']:.2f}, Birth={row['C2_Birth_Rate']:.2f}, "
                      f"Agg={row['C2_Aggression']:.2f}, T={row['C2_Start_Time']:.2f}")
        else:
            title += "\nC2: Mut=0.00, Birth=0.00, Agg=0.00, T=0.00"
        
        plt.title(title)
        plt.xlabel('Bin Index')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize filename
        safe_range = mutation_range.replace(' ', '_').replace('(', '').replace(')', '')
        plot_filename = f"cmp_{clone_type}_{safe_range}_{sample_id}.png"
        plot_path = OUTPUT_DIR / plot_filename
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        # print(f"    Saved plot to {plot_path}")

    # Calculate and print FID scores
    print("\n--- FID Scores ---")
    real_arr = np.array(real_vectors)
    pred_arr = np.array(pred_vectors)
    
    if len(real_arr) > 1:
        try:
            fid_mlp = calculate_fid(real_arr, pred_arr)
            print(f"FID (Real vs MLP): {fid_mlp:.4f}")
            
            if len(meanflow_vectors) == len(real_vectors):
                meanflow_arr = np.array(meanflow_vectors)
                fid_meanflow = calculate_fid(real_arr, meanflow_arr)
                print(f"FID (Real vs MeanFlow): {fid_meanflow:.4f}")
            else:
                print(f"Skipping MeanFlow FID: Sample count mismatch ({len(meanflow_vectors)} vs {len(real_vectors)})")
        except Exception as e:
            print(f"Error calculating FID: {e}")
    else:
        print("Not enough samples to calculate FID (need > 1).")

    # Save predicted data
    pred_df = pd.DataFrame(predicted_data)
    pred_csv_path = OUTPUT_DIR / 'predicted_vaf_data.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Predicted data saved to {pred_csv_path}")

if __name__ == "__main__":
    main()
