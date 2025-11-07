"""
    根据data数据，提取出vaf_wholetumour_0.csv的vaf列并进行排序，最后统计vaf在0-1之间的分布情况。
    vaf区间为 0.00-0.01, 0.01-0.02, ..., 0.99-1.00，共 100 个区间。
"""

data_dir = '/root/data/wja/project/CHESS.cpp/data_original/data'

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    bins = np.arange(0, 1.01, 0.01)  # 0.00 ~ 1.00, step 0.01, ensures last bin includes 1.0

    for folder_path, _, files in tqdm(os.walk(data_dir), desc="Scanning directories"):
        if 'vaf_wholetumour_0.csv' not in files:
            continue

        vaf_file_path = os.path.join(folder_path, 'vaf_wholetumour_0.csv')
        output_file_path = os.path.join(folder_path, 'vaf_distribution.csv')

        df = pd.read_csv(vaf_file_path)
        if 'vaf' not in df.columns:
            print(f"'vaf' column not found in {vaf_file_path}")
            continue

        vaf_values = df['vaf'].dropna().to_numpy()

        hist, _ = np.histogram(vaf_values, bins=bins)

        # build distribution dataframe and add normalized (proportion) column
        total = int(hist.sum())
        distribution_df = pd.DataFrame({
            "vaf_range": [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)],
            "count": hist
        })

        # normalized: proportion of total counts for each bin (0.0 if total is 0)
        if total > 0:
            distribution_df["normalized"] = distribution_df["count"] / total
        else:
            distribution_df["normalized"] = 0.0

        distribution_df.to_csv(output_file_path, index=False)
        print(f"VAF distribution saved to {output_file_path}")
