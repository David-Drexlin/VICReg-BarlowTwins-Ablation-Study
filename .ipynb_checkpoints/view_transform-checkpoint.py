# view_transform.py

import torchvision.transforms as transforms


class ViewTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(),
        ])

    def __call__(self, x):
        # add instances / stack for proposed research question
        x1 = self.transform(x)
        x2 = self.transform(x)
        return (x1, x2)
