import os
from typing import List
import selfies as sf
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class SmilesToSelfies:
    def __init__(self):
        pass

    def transform(self, smiles: List[str]) -> List[str]:
        selfies = []
        failed_smiles = []

        for s in smiles:
            try:
                selfies.append(sf.encoder(s))
            except Exception as e:
                failed_smiles.append(s)

        if failed_smiles:
            print(f"Wrong smiles: {len(failed_smiles)} ")
        
        return selfies

    def inverse_transform(self, selfies: List[str]) -> List[str]:
        return [sf.decoder(s) for s in selfies if s is not None]


class FileToSelfies:
    def __init__(self, smiles_dir: str, selfies_dir: str):
    
        self.smiles_dir = smiles_dir.rstrip('/')  
        self.selfies_dir = selfies_dir.rstrip('/')
        os.makedirs(self.selfies_dir, exist_ok=True)  
    
    def read_smiles(self) -> None:
        smiles_files = sorted(os.listdir(self.smiles_dir))  

        with ProcessPoolExecutor(max_workers=24) as executor:
            list(tqdm(executor.map(self.process_file, smiles_files), total=len(smiles_files)))

    def process_file(self, file_name: str) -> None:
        file_path = os.path.join(self.smiles_dir, file_name)
        df = pd.read_parquet(file_path)
        
        if "smiles" not in df.columns:
            print(f"last column is not smiles in {file_name}")
            return
        
        smiles = df["smiles"].dropna().tolist()  
        selfies = SmilesToSelfies().transform(smiles)

        if selfies:
            output_path = os.path.join(self.selfies_dir, f"selfies_{file_name}")
            self._selfies_to_parquet(selfies, output_path)
    
    def _selfies_to_parquet(self, selfies: List[str], output_path: str) -> None:
        try:
            df = pd.DataFrame({"selfies": selfies})
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"cant write  {output_path}: {e}")
class TxtToSelfies:
    def __init__(self, smiles_path: str, selfies_path: str):
        self.smiles_path = smiles_path
        self.selfies_path = selfies_path
    
    def convert_smiles(self) -> None:
        df = pd.read_csv(self.smiles_path)
    
        smiles = df["smiles"].dropna().tolist()
        
        selfies = SmilesToSelfies().transform(smiles)
        self._selfies_to_txt(self.selfies_path, selfies)
        
    def _selfies_to_txt(self, output_path: str, selfies: List[str]) -> None:
        with open(output_path, "w") as f:
            for s in selfies:
                f.write(f"{s}\n")
    
    
def main():
    smiles_dir = '/data/tzeshinchen/research/dataset/zinc20'
    selfies_dir = '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset'

    # converter = FileToSelfies(smiles_dir, selfies_dir)
    # converter.read_smiles()
    
    smiles_path = '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset/Inhibitor/kinase_in.csv'
    converter = TxtToSelfies(smiles_path, '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset/inhibitor_selfies.txt')
    converter.convert_smiles()
    
    
    

if __name__ == "__main__":
    main()