import pandas as pd
import os

methods = ['pca', 'gaussian', 'ceres']

results = []
for m in methods:
    path = f'statistics/{m}/evaluation_summary.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row['Method'] = m.upper()
        results.append(row)

if results:
    res_df = pd.DataFrame(results)
    cols = ['Method', 'count', 'trans_rmse', 'trans_mean', 'trans_std', 'ang_mean_deg', 'ang_std_deg']
    print(res_df[cols].to_string(index=False))
