import torch
from torch.utils.data import Dataset, DataLoader

class KinaseSelfiesDataset(Dataset):
    def __init__(self, kinase_embeddings, selfies_embeddings, max_length=1536):
        assert len(kinase_embeddings) == len(selfies_embeddings) # pair-wise
        self.kinase_embeddings = kinase_embeddings  # [N, 1024]
        self.selfies_embeddings = selfies_embeddings  # [N, 512]
        self.max_length = max_length

    def __len__(self):
        return len(self.kinase_embeddings)

    def __getitem__(self, idx):
        kinase_emb = self.kinase_embeddings[idx] 
        selfies_emb = self.selfies_embeddings[idx]
        kinase_emb_len = len(kinase_emb)
        combined_tensor = torch.cat([kinase_emb, selfies_emb], dim=0)

        attention_mask = torch.ones(self.max_length) 

        labels = torch.cat([
            torch.full((kinase_emb_len,), -100), 
            selfies_emb
        ])

        return {
            "input_ids": combined_tensor,  
            "attention_mask": attention_mask,
            "labels": labels
        }


if __name__ == "__main__":
    # Test
    kinase_embeddings = [torch.randn(1024) for _ in range(10)]
    selfies_embeddings = [torch.randn(512) for _ in range(10)]
    dataset = KinaseSelfiesDataset(kinase_embeddings, selfies_embeddings)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in dataloader:
        print(batch["input_ids"].shape)  # torch.Size([2, 1536])
        print(batch["attention_mask"].shape)  # torch.Size([2, 1536])
        print(batch["labels"].shape)  # torch.Size([2, 1536])
        break