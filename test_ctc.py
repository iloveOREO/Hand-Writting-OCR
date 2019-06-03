from TextDataset import *
import torch

if __name__ == '__main__':
    ocr_dataset = TextDataset("./data", None)
    ocr_dataset_loader = torch.utils.data.DataLoader(dataset=ocr_dataset,
                                                     batch_size=5,
                                                     shuffle=False,
                                                     collate_fn=alignCollate(imgH=32, imgW=1600))
    for train_iter in ocr_dataset_loader:
        cpu_images, cpu_texts = train_iter
        print(cpu_images.shape)