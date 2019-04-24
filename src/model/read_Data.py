from torch.utils.data import Dataset
import torch
import pickle
import os
import cv2


class TextDataset(Dataset):
    def __init__(self, data_path, transform):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        with open("../data/processed_sentence", "rb") as f:
            self.text = pickle.load(f)
        self.images_names = list(self.text.keys())
        self.charset = """ !"#&'()*+,-./0123456789:;?abcdefghijklmnopqrstuvwxyz\\"""

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        name = self.images_names[idx]
        text = self.text[name]

        img = cv2.imread(os.path.join(self.data_path, name + ".png"))
        seq = self.text_to_seq(text)
        # sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        if self.transform:
            img = self.transform(img)
        return (img, seq)

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.charset.find(c) + 1)
        return seq

if __name__ == '__main__':
    ocr_dataset = TextDataset("../data", None)
    ocr_dataset_loader = torch.utils.data.DataLoader(dataset=TextDataset,
                                                    batch_size=10,
                                                    shuffle=False)
    for img, text in ocr_dataset:
        print(img)
        print(text)
        break
