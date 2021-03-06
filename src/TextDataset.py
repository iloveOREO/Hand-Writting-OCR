from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pickle
import os
# from PIL import Image
import numpy as np
import cv2
from PIL import Image
import pandas as pd

charset = """ !"#&'()*+,-./0123456789:;?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\\"""
alphabet_dict = {charset[i]: i for i in range(len(charset))}


class TextDataset(Dataset):

    # image_data: pd.core.frame.DataFrame

    def __init__(self, data_path, transform=transforms.ToTensor()):
        base_dir = "/home/pdeng/Documents/pickle_data/"
        files = os.listdir(base_dir)
        plk_file = [file for file in files if file.endswith('plk')]
        image_data_chunks = []
        for fn in sorted(plk_file):
            fn = base_dir + fn
            df = pickle.load(open(fn, 'rb'))
            image_data_chunks.append(df)
            self.image_data = pd.concat(image_data_chunks)
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        # TODO GET RIDE OF THIS IN THE FUTURE
        with open("./data/processed_sentence", "rb") as f:
            self.text = pickle.load(f)
        self.images_names = list(self.text.keys())

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):
        '''
        name = self.images_names[idx]
        text = self.text[name].lower()
        image = Image.open(os.path.join(self.data_path, name + ".png")).convert('RGB')
        # TODO ADD TRANSFORM TO SHIFT THE IMAGE
        # image = resize_image(image)
        if self.transform:
            image = self.transform(image)

        seq = self.text_to_seq(text)
        # sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        # print(seq)
        return (image, seq)
        '''
        image, label = self.image_data.iloc[idx][1:]
        image, label, label_size = self.transform(image, label)
        return image, label, label_size

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.charset.find(c) + 1)
        return seq


def transform(image, label):
    image = image / 255
    # SUB MEAN AND DIVIDE THE STD
    image = (image - 0.942532484060557) / 0.15926149044640417
    label_encoded = []
    word_size_count = []
    for word in label:
        word = word.replace("&quot", r'"')
        word = word.replace("&amp", r'&')
        word = word.replace('";', '\"')
        for letter in word:
            label_encoded.append(alphabet_dict[letter])
        word_size_count.append(len(word))
        image = image.astype(np.float32, copy=False)
        image = transforms.ToTensor()(image)
    return image, label_encoded, word_size_count


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


class Collate(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        inputs, labels, labels_count = zip(*batch)
        images = torch.cat([t.unsqueeze(0) for t in inputs], 0)
        labels = torch.cat([torch.IntTensor(t) for t in labels])
        labels_count = torch.cat([torch.IntTensor(t) for t in labels_count])
        # labels_count = torch.cat([torch.Tensor(t) for t in labels])
        return images, labels, labels_count


if __name__ == '__main__':
    ocr_dataset = TextDataset("./data", transform)
    ocr_dataset_loader = torch.utils.data.DataLoader(dataset=ocr_dataset,
                                                     batch_size=4,
                                                     shuffle=False, collate_fn=Collate()
                                                     )
    for i, data in enumerate(ocr_dataset_loader):
        inputs, labels, labels_count = data
        print(inputs.shape)
        print(labels)
        print(labels_count)
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
