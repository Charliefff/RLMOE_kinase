# 階段 1：數據處理與特徵建構

## 目標
準備 kinase inhibitors 數據，並建立 MOE 和 RL 需要的輸入特徵。此外，目標是從 kinase 序列（kinase seq）預測對應的 kinase compound。

### (1) 選擇數據集
你可以使用以下數據庫來獲取 kinase inhibitors：

- [ChEMBL](https://www.ebi.ac.uk/chembl/)
- [BindingDB](https://www.bindingdb.org/)
- [ZINC20](https://zinc20.docking.org/)

### 數據準備步驟
下載 kinase inhibitors，確保包含：

- SMILES（分子結構）
- Binding Affinity (pIC50, Kd, Ki)
- ADMET 性質 (可用 DeepTox 預測)
- Kinase 類別（不同 inhibitors 對應不同的 kinase）
- **Kinase 序列（kinase seq）**

### SMILES 轉換成特徵

- **Graph 表示法**：使用 RDKit & DeepChem 轉換為 molecular graph (可用於 GNN)
- **Fingerprint**：Morgan fingerprint, ECFP
- **分子嵌入 (Molecular Embedding)**：使用 **自行訓練的 BERT embedding model** 來學習分子特徵
- **Kinase Sequence Embedding**：使用 **自行訓練的 BERT embedding model** 解析 kinase seq

### 工具
- **RDKit**（SMILES 解析、特徵轉換）
- **DeepChem**（分子特徵建構）
- **自行訓練的 BERT 模型**（生成分子與 kinase seq 的嵌入）

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
import torch
from transformers import BertModel, BertTokenizer

# 加載自行訓練的 BERT 模型
tokenizer = BertTokenizer.from_pretrained("path/to/custom/bert")
model = BertModel.from_pretrained("path/to/custom/bert")

def get_bert_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 讀取 SMILES
smiles = "CCOCC(=O)N1CCC(CN2CCOCC2)CC1"
mol = Chem.MolFromSmiles(smiles)

# 轉換為 Fingerprint
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# 用 DeepChem 轉 Graph
featurizer = dc.feat.ConvMolFeaturizer()
graph = featurizer.featurize(mol)

# 獲取 BERT 特徵
kinase_seq = "MAEKLLRQ"
kinase_embedding = get_bert_embedding(kinase_seq)
```

# 階段 2：建立 MOE (Mixture of Experts)

## 目標
構建 MOE 模型，使不同的 Experts 對應不同的 kinase inhibitor 類別，並根據 kinase seq 預測對應的 kinase compound。

### (1) MOE 結構
MOE 由：

- **Gating Network (G)** 負責選擇最佳 Expert
- **Expert Networks (E1, E2, ..., En)** 各負責不同 kinase inhibitors
- **Kinase Sequence Encoder** 提取 kinase 序列的特徵並與 MOE 融合

### 實作方式

- **Gating Network**：Transformer / GNN-based Selector
- **Expert Models**：
  - CNN/GNN for Graph Input
  - Transformer for Sequence Input
  - MLP for Fingerprint Input
  - **自行訓練的 BERT 模型** for Kinase Seq

```python
import torch.nn as nn

class KinaseSeqEncoder(nn.Module):
    def __init__(self, bert_model_path):
        super(KinaseSeqEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
    
    def forward(self, x):
        return self.bert(x).last_hidden_state.mean(dim=1)
```

### MOE 主要運行方式

1. **Gating Network 選擇最佳 Expert**
2. **使用選中的 Experts 計算結果**
3. **融合 kinase seq 與 molecule 特徵來預測 kinase compound**

```python
class MOEModel(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, bert_model_path):
        super(MOEModel, self).__init__()
        self.gating = GatingNetwork(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.kinase_encoder = KinaseSeqEncoder(bert_model_path)

    def forward(self, x, kinase_seq):
        kinase_feature = self.kinase_encoder(kinase_seq)
        gate_weights = self.gating(kinase_feature)  # 根據 kinase seq 選擇 Expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=-1)
```

# 總結

1. **處理 kinase inhibitor 數據 (SMILES → 分子特徵 + kinase seq)**
2. **建立 MOE (Gating Network + Experts + Kinase Seq Encoder)**
3. **用 RL 來微調 MOE (DQN / PPO / GFlowNet)**
4. **使用自行訓練的 BERT 來進行分子與 kinase seq 的嵌入特徵學習**
5. **評估與優化 (Docking 測試, ADMET 分析)**

這樣可以確保 MOE 分子設計與 RL 調優的最佳效果，並且可以根據 kinase seq 預測對應的 kinase compound。你有想要優先實作哪個部分嗎？
