# view_transform.py

import torchvision.transforms as transforms


class ViewTransform(object):
    def __init__(self, num):
        self.num = num
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(),
        ])

    def __call__(self, x):
        # add instances / stack for proposed research question
        transforms = []
        for i in range(self.num):
            transforms.append(self.transform(x))
        return transforms
