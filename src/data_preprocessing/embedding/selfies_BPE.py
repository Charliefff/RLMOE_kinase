from ape_tokenizer import APETokenizer

def build_tokenizer(path, output_path):

    tokenizer = APETokenizer()
    print("Training tokenizer...")
    with open(path, "r") as f:
        selfies_data = [line.strip() for line in f.readlines()]
        
    tokenizer.train(selfies_data, max_vocab_size=10000, 
                    min_freq_for_merge=50, 
                    save_checkpoint=True, 
                    checkpoint_path="./checkpoints")
    tokenizer.save_vocabulary(output_path)
    return


def main():
    path = '/data/tzeshinchen/RLMOE_kinase_inhibitor/dataset/extracted_file.txt'
    output_path = "trained_vocabulary.json"

    print("Vocab file found, building tokenizer")
    build_tokenizer(path, output_path)
    print("Tokenizer built")

if __name__ == '__main__':
 
    main()