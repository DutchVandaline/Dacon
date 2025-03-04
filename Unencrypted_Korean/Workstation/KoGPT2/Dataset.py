from torch.utils.data import Dataset

class ObfuscationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        obf_text = row["clean_output"]
        norm_text = row["output"]

        prompt = "Obfuscated: " + obf_text + " Normal:"
        full_text = prompt + " " + norm_text

        encoding = self.tokenizer(
            full_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        prompt_encoding = self.tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_encoding["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
