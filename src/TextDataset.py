from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
import os
# from PIL import Image
import numpy as np
import cv2
from PIL import Image


class TextDataset(Dataset):

    def __init__(self, data_path, transform=transforms.ToTensor()):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        with open("./data/processed_sentence.pickle", "rb") as f:
            self.text = pickle.load(f)
        self.images_names = list(self.text.keys())
        self.charset = """ !"#&'()*+,-./0123456789:;?abcdefghijklmnopqrstuvwxyz\\"""

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        name = self.images_names[idx]
        text = self.text[name].lower()

        image = Image.open(os.path.join(self.data_path, name + ".png")).convert('RGB')

        # image = resize_image(image)
        if self.transform:
            image = self.transform(image)

        seq = self.text_to_seq(text)
        # sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        # print(seq)
        return (image, seq)

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.charset.find(c) + 1)
        return seq


def resize_image(image, desired_size=[80, 1600]):
    size = image.shape[:2]
    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = image[0][0]
    if color.any() < 230:
        color = 230
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))

    image[image > 230] = 255
    return image


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=80, imgW=1600, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


if __name__ == '__main__':
    ocr_dataset = TextDataset("./data", None)
    ocr_dataset_loader = torch.utils.data.DataLoader(dataset=ocr_dataset,
                                                     batch_size=4,
                                                     shuffle=False,
                                                     collate_fn=alignCollate(imgH=80, imgW=1600, keep_ratio=True))
    for i, data in enumerate(ocr_dataset_loader):
        inputs, labels = data
        print(inputs.shape)
        print(len(labels))
        break

# def text_collate(batch):
#     img = list()
#     seq = list()
#     seq_len = list()
#     for sample in batch:
#         img.append(torch.from_numpy(sample["img"].transpose((2, 0, 1))).float())
#         seq.extend(sample["seq"])
#         seq_len.append(sample["seq_len"])
#     img = torch.stack(img)
#     seq = torch.Tensor(seq).int()
#     seq_len = torch.Tensor(seq_len).int()
#     batch = {"img": img, "seq": seq, "seq_len": seq_len}
#     return batch
