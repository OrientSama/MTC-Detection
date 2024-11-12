from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, mask_images_path: list, transform=None, norm=None):
        self.images_path = images_path
        self.images_class = images_class
        self.mask_images_path = mask_images_path
        self.transform = transform
        self.totensor = transforms.ToTensor()
        self.norm = norm

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        mask = Image.open(self.mask_images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img, mask = self.totensor(img), self.totensor(mask)
            img_mask = self.transform(torch.cat([img, mask], dim=0))
            img, mask = torch.split(img_mask,[3, 1], dim=0)
            img = self.norm(img)

        return img, label, mask

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels, masks = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, masks


class TestDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, transform=None, norm=None):
        self.images_path = images_path
        self.transform = transform
        self.totensor = transforms.ToTensor()
        self.norm = norm

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.totensor(img)
            img = self.transform(img)
            img = self.norm(img)

        return img

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # images = tuple(batch)
        images = torch.stack(batch, dim=0)
        return images
