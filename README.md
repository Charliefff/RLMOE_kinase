# **Protein-Compound MOE 訓練計畫**

## **階段 1：數據處理與 Protein-Compound GPT 訓練**
### **目標**
- 訓練 **Protein-Compound GPT 模型**，學習蛋白質序列與化合物結構的關聯。
- 生成的分子表示為 **SELFIES**，確保可行性。
- 這個 GPT 將成為 **MOE（Mixture of Experts）** 的基礎模型。

### **數據準備**
- **數據庫來源**：
  - ChEMBL / BindingDB / ZINC20
- **必備數據**：
  - **蛋白質序列（FASTA）**
  - **SMILES（化合物結構）**
  - **Binding Affinity（pIC50, Kd, Ki）**
  - **ADMET 屬性（可用 DeepTox 預測）**

### **特徵工程**
- **蛋白質嵌入（Protein Embedding）**：使用 **ProtT5 / ESM2** 訓練蛋白質的 embedding。
- **分子嵌入（Compound Embedding）**：
  - SMILES 轉 **SELFIES**
  - **RDKit / DeepChem** 提取 fingerprint。
- **數據格式**： `protein | compound`，作為 GPT 的輸入。

### **訓練策略**
- **使用 GPT 訓練 `Protein → Compound` 生成能力**
- **目標：讓 GPT 能根據蛋白質序列生成合理的分子結構（SELFIES）**

---

## **階段 2：MOE 設計與 RL 訓練**
### **目標**
- **將訓練好的 Protein-Compound GPT，複製 8 份，形成 MOE 專家模型**。
- **使用 RL 訓練 Gating Network，使其學習選擇最適合的 Expert**。
- **不同 Experts 在 RL 訓練後，專精於不同 kinase families**。

### **MOE 結構**
- **共享 GPT 主幹**（Transformer 層）
- **獨立的 8 個 Expert（FC 層）**
- **Gating Network（Transformer-based 或 GNN-based），決定使用哪些 Experts**

### **訓練策略**
- **Gating Network 訓練**
  - **輸入**：蛋白質序列（kinase）
  - **輸出**：選擇適合的 GPT Experts
  - **強化學習（PPO / DQN）** 優化 Gating Network
- **Expert RL 訓練**
  - 透過 reward（**QED, SA Score, Docking Score, ADMET**）
  - **讓不同 Experts 針對特定 kinase 最佳化分子生成**

---

## **階段 3：模型優化與測試**
### **目標**
- **微調 RL 訓練，確保不同 Experts 生成最適應 kinase 的化合物**。
- **評估分子質量（Binding Affinity, QED, ADMET）**。
- **最終輸出 kinase-specific compound，進行 Docking 測試。**

### **模型評估**
- **生成分子合成可行性（SA Score, QED）**
- **與 kinase 結合的親和力（Docking Score）**
- **ADMET 屬性分析**
- **與真實藥物數據對比（BindingDB / ChEMBL）**

---

## **總結**
1. **先訓練 Protein-Compound GPT，學習一般蛋白質與化合物的關係。**
2. **複製 8 份 GPT，建立 MOE，透過 RL 訓練不同 Experts。**
3. **使用 Gating Network 動態選擇最適合的 GPT Expert。**
4. **微調模型，確保生成的分子符合藥物開發標準。**

