import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Docstring for results.snpe.run
根据README.md中的5个sample的参数，进行输入程序，保存结果到对应的文件夹，然后得到vaf，最后可视化
"""
BINARY_NAME = 'cancer_gillespie_simulation_no_display'

samples_params = {
    'sample1': {
        'mutation_rates': 88.4965,
        'selective_advantages': 11.2006,
        'death_rates': 0.0000,
        'aggressions': 0.4507,
        'time_to_new_clone': 6.1861
    },
    'sample2': {
        'mutation_rates': 55.3703,
        'selective_advantages': 10.6806,
        'death_rates': 0.0000,
        'aggressions': 0.1392,
        'time_to_new_clone': 7.1486
    },
    'sample3': {
        'mutation_rates': 18.5367,
        'selective_advantages': 10.0505,
        'death_rates': 0.0000,
        'aggressions': 0.7008,
        'time_to_new_clone': 6.8931
    },
    'sample4': {
        'mutation_rates': 10.1810,
        'selective_advantages': 10.0605,
        'death_rates': 0.0000,
        'aggressions': 0.6987,
        'time_to_new_clone': 6.8566
    },
    'sample5': {
        'mutation_rates': 33.9557,
        'selective_advantages': 11.0951,
        'death_rates': 0.0000,
        'aggressions': 0.0739,
        'time_to_new_clone': 7.1632
    },
}

def _format_values(val):
    return str(val)

def _build_command(binary_path, params, output_dir):
    command = [
        str(binary_path),
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

def process_vaf(output_dir):
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
        plt.plot(range(len(distribution_df)), distribution_df['normalized'], color='skyblue', linewidth=2)
        plt.xticks([])
        plt.xlabel('VAF Range')
        plt.ylabel('Normalized Frequency')
        plt.title('VAF Distribution')
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
        process_vaf(output_dir)

if __name__ == "__main__":
    main()
