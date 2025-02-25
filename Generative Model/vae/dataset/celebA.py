import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as tf
from   torchvision.datasets import ImageFolder

class CelebADataset(data.Dataset):
    def __init__(self, args, is_train=False):
        super().__init__()
        #-----------------basic parameters------------------#
        self.img_size = args.img_size
        self.is_train = is_train
        self.pixel_mean = [0.0, 0.0, 0.0]
        self.pixel_std = [1.0, 1.0, 1.0]
        print("Image size: {}".format(self.img_size))
        print("Pixel mean: {}".format(self.pixel_mean))
        print("Pixel std: {}".format(self.pixel_std))
        #-----------------dataset and transforms-------------#
        self.transform = self.build_transforms()
        self.dataset = ImageFolder(root=self.data_path, transform=self.transform)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target
    
    def pull_image(self, index):
        # load data
        image, target = self.dataset[index]

        # denormalize image
        image = image.permute(1, 2, 0).numpy()
        image = (image * self.pixel_std + self.pixel_mean) * 255.0
        image = image.astype(np.uint8)
        image = image.copy()
        return image, target

    def build_transform(self,):
        if self.is_train:
            transforms = tf.Compose([
                            tf.Resize([self.img_size, self.img_size]),
                            tf.RandomHorizontalFlip(0.5),
                            tf.ToTensor(),
                            tf.Normalize(self.pixel_mean, self.pixel_std)])
            
        else:
            transforms = tf.Compose([
                            tf.Resize([self.img_size, self.img_size]),
                            tf.ToTensor(),
                            tf.Normalize(self.pixel_mean, self.pixel_std)])

        return transforms
    
if __name__ == '__main__':
    import cv2
    import argparse
    parser = argparse.ArgumentParser(description='CelebA dataset')
    parser.add_argument('--root', default='path/to/celebA', help='path to celebA dataset')
    args = parser.parse_args()

    # Dataset
    dataset = CelebADataset(args, is_train=True)
    print("Dataset size: {}".format(len(dataset)))

    for i in range(1000):
        image, target = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.0).astype(np.uint8)

        image = image[..., (2, 1, 0)]

        cv2.imshow('image', image)
        cv2.waitKey(0)