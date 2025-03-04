import pandas as pd

class CharTokenizer:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        hangul_list = df["Hangul Combinations"].tolist()

        special_tokens = [" ", ".", ",", "!", "?"]
        hangul_list.extend(special_tokens)

        self.vocab = {char: idx for idx, char in enumerate(hangul_list)}

    def tokenize(self, text):
        return [self.vocab[char] for char in text if char in self.vocab]

    def detokenize(self, token_ids):
        id_to_char = {idx: char for char, idx in self.vocab.items()}
        return "".join(id_to_char[idx] for idx in token_ids if idx in id_to_char)
