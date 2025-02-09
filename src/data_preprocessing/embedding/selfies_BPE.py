from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex


def main():
    path = '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset/selfies.txt'
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(r"(\[[^\]]+\])"), behavior="isolated")
    trainer = trainers.BpeTrainer(
        vocab_size=3000, 
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train([path], trainer)

    # 儲存 Tokenizer
    tokenizer.save("./selfies_bpe.json")

    
if __name__ == '__main__':
    main()