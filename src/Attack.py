import random
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms


class BackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with pixel-trigger backdoor poisoning.
    Inserts a colored square patch (trigger) into a fraction of images and changes their labels.
    Supports multiple target labels.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, trigger_size=3, trigger_location='bottom_right',
                 trigger_color=(1.0, 0.0, 0.0), target_labels=(0,)):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.trigger_color = trigger_color
        # Ensure tuple of labels
        self.target_labels = tuple(target_labels)

        num_samples = len(self.cifar)
        num_poison = int(self.poison_rate * num_samples)
        poisoned_idxs = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned_idxs)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        label = _
        if idx in self.poisoned_set:
            img_np = np.array(img).astype(np.float32) / 255.0
            h, w, _ = img_np.shape
            ts = self.trigger_size
            loc = {
                'bottom_right': (w-ts, h-ts),
                'bottom_left': (0, h-ts),
                'top_right': (w-ts, 0),
                'top_left': (0, 0)
            }.get(self.trigger_location)
            if loc is None:
                raise ValueError(f"Unknown trigger_location: {self.trigger_location}")
            x_start, y_start = loc
            img_np[y_start:y_start+ts, x_start:x_start+ts, :] = np.array(self.trigger_color)[None, None, :]
            img = transforms.ToPILImage()(np.clip(img_np, 0, 1))
            # Randomly choose one target label if multiple
            label = random.choice(self.target_labels)

        if self.transform:
            img = self.transform(img)
        return img, label


class SemanticBackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with semantic backdoor poisoning.
    Overlays a striped pattern across the image and changes labels for poisoned samples.
    Supports multiple target labels.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, stripe_width=4, alpha=0.3,
                 stripe_orientation='vertical', target_labels=(0,)):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.stripe_width = stripe_width
        self.alpha = alpha
        self.stripe_orientation = stripe_orientation
        self.target_labels = tuple(target_labels)

        num_samples = len(self.cifar)
        num_poison = int(self.poison_rate * num_samples)
        poisoned_idxs = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned_idxs)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, _ = self.cifar[idx]
        label = _
        if idx in self.poisoned_set:
            img_np = np.array(img).astype(np.float32) / 255.0
            h, w, c = img_np.shape
            mask = np.zeros((h, w), dtype=np.float32)
            if self.stripe_orientation == 'vertical':
                for x in range(0, w, 2 * self.stripe_width):
                    mask[:, x:x+self.stripe_width] = 1.0
            else:
                for y in range(0, h, 2 * self.stripe_width):
                    mask[y:y+self.stripe_width, :] = 1.0
            mask = mask[:, :, None]
            stripe_color = np.ones((h, w, c), dtype=np.float32)
            img_np = img_np * (1 - self.alpha * mask) + stripe_color * (self.alpha * mask)
            img = transforms.ToPILImage()(np.clip(img_np, 0, 1))
            label = random.choice(self.target_labels)

        if self.transform:
            img = self.transform(img)
        return img, label
