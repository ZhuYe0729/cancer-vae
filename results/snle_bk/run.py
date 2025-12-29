import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Docstring for results.snpe.run

"""
BINARY_NAME = 'cancer_gillespie_simulation_no_display'
TEST_RESULTS_PATH = '/root/wja/wja/project/tsnpe_neurips/my/test_results_snle.npz'

samples_params = {
    'sample1': {
        'mutation_rates': 52.2979,
        'selective_advantages': 10.4714,
        'death_rates': 0.4984,
        'aggressions': 0.5030,
        'time_to_new_clone': 7.4865
    },
    'sample2': {
        'mutation_rates': 69.9655,
        'selective_advantages': 12.4773,
        'death_rates': 0.0000,
        'aggressions': 0.8928,
        'time_to_new_clone': 4.0098
    },
    'sample3': {
        'mutation_rates': 47.5544,
        'selective_advantages': 10.3280,
        'death_rates': 0.4825,
        'aggressions': 0.4816,
        'time_to_new_clone': 7.8071
    },
    'sample4': {
        'mutation_rates': 52.4797,
        'selective_advantages': 10.0157,
        'death_rates': 0.5214,
        'aggressions': 0.4938,
        'time_to_new_clone': 6.8516
    },
    'sample5': {
        'mutation_rates': 49.6598,
        'selective_advantages': 10.9923,
        'death_rates': 0.5266,
        'aggressions': 0.4895,
        'time_to_new_clone': 7.1500
    },
}

def _format_values(val):
    return str(val)

def _build_command(binary_path, params, output_dir):
    command = [
        str(binary_path),
        '-i', '0', # Set simulation ID to 0 to match expected output filename
        '-M','1000',
        '-m', _format_values(params['mutation_rates']),
        '-b', _format_values(params['selective_advantages']),
        # '-d', _format_values(params['death_rates']),
        '-d', '0',
        '-r', _format_values(params['aggressions']),
        '-t', _format_values(params['time_to_new_clone']),
        '-o', str(output_dir) + '/',
    ]
    return command


def _run_command(command):
    print(f"Running command: {' '.join(command)}")
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout

def process_vaf(output_dir, true_vaf_dist=None):
    # Look for vaf_wholetumour_0.csv as we set -i 0
    vaf_file_path = os.path.join(output_dir, 'vaf_wholetumour_0.csv')
    
    if not os.path.exists(vaf_file_path):
        print(f"VAF file not found: {vaf_file_path}")
        return

    output_file_path = os.path.join(output_dir, 'vaf_distribution.csv')
    
    bins = np.arange(0, 1.01, 0.01)  # 0.00 ~ 1.00, step 0.01
    
    try:
        df = pd.read_csv(vaf_file_path)
        if 'vaf' not in df.columns:
            print(f"'vaf' column not found in {vaf_file_path}")
            return

        vaf_values = df['vaf'].dropna().to_numpy()

        hist, _ = np.histogram(vaf_values, bins=bins)

        total = int(hist.sum())
        distribution_df = pd.DataFrame({
            "vaf_range": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
            "count": hist
        })

        if total > 0:
            distribution_df["normalized"] = distribution_df["count"] / total
        else:
            distribution_df["normalized"] = 0.0

        distribution_df.to_csv(output_file_path, index=False)
        print(f"VAF distribution saved to {output_file_path}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(distribution_df)), distribution_df['normalized'], color='skyblue', linewidth=2, label='Predicted')
        
        if true_vaf_dist is not None:
            # Normalize true_vaf_dist if it's not already normalized
            true_vaf_norm = true_vaf_dist
            if np.sum(true_vaf_dist) > 1.5: # Heuristic check if it's counts
                 true_vaf_norm = true_vaf_dist / np.sum(true_vaf_dist)
            
            plt.plot(range(len(true_vaf_norm)), true_vaf_norm, color='orange', linewidth=2, linestyle='--', label='True')

        plt.xticks([])
        plt.xlabel('VAF Range')
        plt.ylabel('Normalized Frequency')
        plt.title('VAF Distribution')
        plt.legend()
        plt.tight_layout()
        plt_path = os.path.join(output_dir, 'vaf_distribution.png')
        plt.savefig(plt_path)
        plt.close()
        print(f"VAF distribution plot saved to {plt_path}")
        
    except Exception as e:
        print(f"Error processing VAF for {output_dir}: {e}")

def main():
    # Determine binary path
    # Assuming run.py is in results/snpe/
    # Binary is in project root: ../../cancer_gillespie_simulation_no_display
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    binary_path = os.path.join(project_root, BINARY_NAME)
    
    if not os.path.exists(binary_path):
        print(f"Binary not found at {binary_path}")
        return

    # Load test results
    true_vafs = None
    if os.path.exists(TEST_RESULTS_PATH):
        try:
            data = np.load(TEST_RESULTS_PATH)
            true_vafs = data['true_vaf'] # Shape (N, 100)
            print(f"Loaded true VAFs from {TEST_RESULTS_PATH}")
        except Exception as e:
            print(f"Failed to load {TEST_RESULTS_PATH}: {e}")
    else:
        print(f"Test results file not found at {TEST_RESULTS_PATH}")

    sample_indices = {
        'sample1': 0,
        'sample2': 1,
        'sample3': 2,
        'sample4': 3,
        'sample5': 4,
    }

    for sample_name, params in samples_params.items():
        print(f"Processing {sample_name}...")
        output_dir = os.path.join(script_dir, sample_name)
        os.makedirs(output_dir, exist_ok=True)
        
        command = _build_command(binary_path, params, output_dir)
        ret_code, stdout = _run_command(command)
        
        if ret_code != 0:
            print(f"Simulation failed for {sample_name}")
            print(stdout)
            continue
        
        print(f"Simulation finished for {sample_name}")
        
        # Get true vaf for this sample
        current_true_vaf = None
        if true_vafs is not None:
            idx = sample_indices.get(sample_name)
            if idx is not None and idx < len(true_vafs):
                current_true_vaf = true_vafs[idx]
        
        process_vaf(output_dir, true_vaf_dist=current_true_vaf)

if __name__ == "__main__":
    main()
