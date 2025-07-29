import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms


def parse_mapping(mapping_str):
    """
    Parse mapping string e.g. '0:5,1:3' into dict {0:5,1:3}.
    """
    mapping = {}
    for pair in mapping_str.split(','):
        if not pair:
            continue
        orig, tgt = pair.split(':')
        mapping[int(orig)] = int(tgt)
    return mapping


class BackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with pixel-trigger backdoor poisoning.
    Inserts a colored square patch into a fraction of images and remaps labels by user-defined rules.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, trigger_size=3, trigger_location='bottom_right',
                 trigger_color=(1.0, 0.0, 0.0), label_mapping=None):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.trigger_color = trigger_color
        self.label_mapping = label_mapping or {}

        num_samples = len(self.cifar)
        num_poison = int(self.poison_rate * num_samples)
        poisoned_idxs = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned_idxs)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, orig_label = self.cifar[idx]
        label = orig_label
        if idx in self.poisoned_set:
            # apply pixel trigger
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

            # remap label according to provided rules
            label = self.label_mapping.get(orig_label, orig_label)

        if self.transform:
            img = self.transform(img)
        return img, label


class SemanticBackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with semantic backdoor poisoning.
    Overlays a striped pattern across the image and remaps labels by user-defined rules.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, stripe_width=4, alpha=0.3,
                 stripe_orientation='vertical', label_mapping=None):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.stripe_width = stripe_width
        self.alpha = alpha
        self.stripe_orientation = stripe_orientation
        self.label_mapping = label_mapping or {}

        num_samples = len(self.cifar)
        num_poison = int(self.poison_rate * num_samples)
        poisoned_idxs = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned_idxs)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img, orig_label = self.cifar[idx]
        label = orig_label
        if idx in self.poisoned_set:
            # apply semantic trigger
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

            # remap label according to provided rules
            label = self.label_mapping.get(orig_label, orig_label)

        if self.transform:
            img = self.transform(img)
        return img, label


class BackdoorMNIST(Dataset):
    """
    MNIST dataset with pixel-trigger backdoor poisoning.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, trigger_size=3, trigger_location='bottom_right',
                 trigger_value=1.0, label_mapping=None):
        self.data = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.trigger_value = trigger_value
        self.label_mapping = label_mapping or {}
        num_samples = len(self.data)
        num_poison = int(self.poison_rate * num_samples)
        poisoned = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, orig = self.data[idx]
        arr = np.array(img).astype(np.float32)/255.0
        label = orig
        if idx in self.poisoned_set:
            h,w = arr.shape
            ts = self.trigger_size
            locs = {
                'bottom_right': (w-ts, h-ts), 'bottom_left': (0,h-ts),
                'top_right': (w-ts,0), 'top_left': (0,0)
            }
            x,y = locs[self.trigger_location]
            arr[y:y+ts,x:x+ts] = self.trigger_value
            img = transforms.ToPILImage()(np.clip(arr,0,1))
            label = self.label_mapping.get(orig, orig)
        if self.transform: img = self.transform(img)
        return img, label


class SemanticBackdoorMNIST(Dataset):
    """
    MNIST dataset with semantic backdoor poisoning.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, stripe_width=4, alpha=0.3,
                 stripe_orientation='vertical', label_mapping=None):
        self.data = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.stripe_width = stripe_width
        self.alpha = alpha
        self.stripe_orientation = stripe_orientation
        self.label_mapping = label_mapping or {}
        num_samples = len(self.data)
        num_poison = int(self.poison_rate * num_samples)
        poisoned = np.random.choice(num_samples, num_poison, replace=False)
        self.poisoned_set = set(poisoned)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, orig = self.data[idx]
        arr = np.array(img).astype(np.float32)/255.0
        label = orig
        if idx in self.poisoned_set:
            h,w = arr.shape
            mask = np.zeros((h,w),dtype=np.float32)
            if self.stripe_orientation=='vertical':
                for x in range(0,w,2*self.stripe_width): mask[:,x:x+self.stripe_width]=1
            else:
                for y in range(0,h,2*self.stripe_width): mask[y:y+self.stripe_width,:]=1
            arr = arr*(1-self.alpha*mask) + 1.0*(self.alpha*mask)
            img = transforms.ToPILImage()(np.clip(arr,0,1))
            label = self.label_mapping.get(orig, orig)
        if self.transform: img = self.transform(img)
        return img, label
