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


def _get_targets(ds):
    # tương thích nhiều phiên bản torchvision
    for name in ["targets", "train_labels", "test_labels"]:
        if hasattr(ds, name):
            return np.array(getattr(ds, name))
    raise AttributeError("Cannot find targets in dataset")


def _sample_poison_indices(labels, label_mapping, poison_rate, seed=None):
    # chỉ lấy ứng viên là các mẫu có nhãn gốc thuộc keys của mapping
    if not label_mapping:
        return set()
    src_labels = np.array(list(label_mapping.keys()))
    candidates = np.where(np.isin(labels, src_labels))[0]
    rng = np.random.default_rng(seed)
    num_poison = int(round(poison_rate * len(candidates)))
    poisoned_idxs = rng.choice(candidates, num_poison, replace=False) if num_poison > 0 else np.array([], dtype=int)
    return set(map(int, poisoned_idxs))


class BackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with pixel-trigger backdoor poisoning.
    Inserts a colored square patch into a fraction of images and remaps labels by user-defined rules.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, trigger_size=3, trigger_location='bottom_right',
                 trigger_color=(1.0, 0.0, 0.0), label_mapping=None, seed=None):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.trigger_color = trigger_color
        self.label_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}

        labels = _get_targets(self.cifar)
        self.poisoned_set = _sample_poison_indices(labels, self.label_mapping, poison_rate, seed)

    def __len__(self): return len(self.cifar)

    def __getitem__(self, idx):
        img, orig_label = self.cifar[idx]
        label = orig_label
        if idx in self.poisoned_set:
            # chỉ xảy ra khi orig_label ∈ label_mapping
            img_np = np.array(img).astype(np.float32) / 255.0
            h, w, _ = img_np.shape
            ts = self.trigger_size
            loc = {
                'bottom_right': (w-ts, h-ts), 'bottom_left': (0, h-ts),
                'top_right': (w-ts, 0), 'top_left': (0, 0)
            }[self.trigger_location]
            x_start, y_start = loc
            img_np[y_start:y_start+ts, x_start:x_start+ts, :] = np.array(self.trigger_color, dtype=np.float32)[None,None,:]
            img = transforms.ToPILImage()(np.clip(img_np, 0, 1))
            label = self.label_mapping[orig_label]

        if self.transform: img = self.transform(img)
        return img, label


class SemanticBackdoorCIFAR10(Dataset):
    """
    CIFAR10 dataset with semantic backdoor poisoning.
    Overlays a striped pattern across the image and remaps labels by user-defined rules.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, stripe_width=4, alpha=0.3,
                 stripe_orientation='vertical', label_mapping=None,
                 stripe_color=(1.0, 1.0, 1.0), seed=None):
        self.cifar = CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = float(poison_rate)
        self.stripe_width = int(stripe_width)
        self.alpha = float(alpha)
        self.stripe_orientation = stripe_orientation
        self.label_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}
        self.stripe_color = tuple(float(x) for x in stripe_color)
        self.seed = seed

        # checks
        assert self.stripe_orientation in ("vertical", "horizontal")
        assert self.stripe_width > 0
        assert 0.0 <= self.alpha <= 1.0
        assert len(self.stripe_color) == 3

        labels = _get_targets(self.cifar)
        self.poisoned_set = _sample_poison_indices(labels, self.label_mapping, self.poison_rate, seed)

    def __len__(self): return len(self.cifar)

    def __getitem__(self, idx):
        img, orig_label = self.cifar[idx]
        label = orig_label

        if idx in self.poisoned_set:
            # tạo sọc
            img_np = np.array(img, dtype=np.float32) / 255.0  # HWC
            h, w, c = img_np.shape
            mask = np.zeros((h, w), dtype=np.float32)
            if self.stripe_orientation == 'vertical':
                for x in range(0, w, 2 * self.stripe_width):
                    mask[:, x:x + self.stripe_width] = 1.0
            else:
                for y in range(0, h, 2 * self.stripe_width):
                    mask[y:y + self.stripe_width, :] = 1.0
            mask = mask[:, :, None]  # HWC, 1 channel
            color = np.array(self.stripe_color, dtype=np.float32).reshape(1, 1, 3)
            img_np = img_np * (1.0 - self.alpha * mask) + color * (self.alpha * mask)
            img = transforms.ToPILImage()(np.clip(img_np, 0.0, 1.0))

            # đổi nhãn (chắc chắn tồn tại)
            label = self.label_mapping[orig_label]

        if self.transform:
            img = self.transform(img)
        return img, label


class BackdoorMNIST(Dataset):
    """
    MNIST dataset with pixel-trigger backdoor poisoning.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, trigger_size=3, trigger_location='bottom_right',
                 trigger_value=1.0, label_mapping=None, seed=None):
        self.data = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.trigger_value = float(trigger_value)
        self.label_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}

        labels = _get_targets(self.data)
        self.poisoned_set = _sample_poison_indices(labels, self.label_mapping, poison_rate, seed)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, orig = self.data[idx]
        label = orig
        if idx in self.poisoned_set:
            arr = np.array(img).astype(np.float32) / 255.0
            h, w = arr.shape
            ts = self.trigger_size
            locs = {
                'bottom_right': (w-ts, h-ts), 'bottom_left': (0, h-ts),
                'top_right': (w-ts, 0), 'top_left': (0, 0)
            }
            x, y = locs[self.trigger_location]
            arr[y:y+ts, x:x+ts] = self.trigger_value
            img = transforms.ToPILImage()(np.clip(arr, 0, 1))
            label = self.label_mapping[orig]  # chắc chắn có

        if self.transform: img = self.transform(img)
        return img, label


class SemanticBackdoorMNIST(Dataset):
    """
    MNIST dataset with semantic backdoor poisoning.
    """
    def __init__(self, root, train=True, transform=None, download=True,
                 poison_rate=0.1, stripe_width=4, alpha=0.3,
                 stripe_orientation='vertical', label_mapping=None, seed=None):
        self.data = MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.poison_rate = float(poison_rate)
        self.stripe_width = int(stripe_width)
        self.alpha = float(alpha)
        self.stripe_orientation = stripe_orientation
        self.label_mapping = {int(k): int(v) for k, v in (label_mapping or {}).items()}
        self.seed = seed

        # checks
        assert self.stripe_orientation in ("vertical", "horizontal")
        assert self.stripe_width > 0
        assert 0.0 <= self.alpha <= 1.0

        labels = _get_targets(self.data)
        self.poisoned_set = _sample_poison_indices(labels, self.label_mapping, self.poison_rate, seed)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, orig = self.data[idx]
        label = orig

        if idx in self.poisoned_set:
            arr = np.array(img, dtype=np.float32) / 255.0  # HW
            h, w = arr.shape
            mask = np.zeros((h, w), dtype=np.float32)
            if self.stripe_orientation == 'vertical':
                for x in range(0, w, 2 * self.stripe_width):
                    mask[:, x:x + self.stripe_width] = 1.0
            else:
                for y in range(0, h, 2 * self.stripe_width):
                    mask[y:y + self.stripe_width, :] = 1.0
            arr = arr * (1.0 - self.alpha * mask) + 1.0 * (self.alpha * mask)  # stripe_value=1.0
            img = transforms.ToPILImage()(np.clip(arr, 0.0, 1.0))

            label = self.label_mapping[orig]

        if self.transform:
            img = self.transform(img)
        return img, label
