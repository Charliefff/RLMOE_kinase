import os
import pandas as pd
from tqdm import tqdm
selfies_file = '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset'
selfies_path = os.listdir(selfies_file)
with open('selfies.txt', 'w') as f:
    for i in tqdm(selfies_path):
        df = pd.read_parquet(f'{selfies_file}/{i}')
        for j in df['selfies']:
            f.write(j + '\n')
        print(f'{i} is done')
