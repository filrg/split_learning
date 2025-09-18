import math
import numpy as np
import torchvision
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from torchvision import transforms

from src.model import *
from src.Attack import _get_targets, parse_mapping
from src.Attack import BackdoorCIFAR10, SemanticBackdoorCIFAR10, BackdoorMNIST, SemanticBackdoorMNIST


class WithIndex(Dataset):
    """Trả thêm chỉ số idx -> (x, y, idx) để nhận diện poisoned mẫu nào."""

    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, idx


@torch.no_grad()
def test(model_name, data_name, state_dict_full, logger):
    if data_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif data_name == "FASHION_MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    elif data_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Data name '{data_name}' is not valid.")

    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if 'MNIST' in data_name:
        klass = globals()[f'{model_name}_MNIST']
    else:
        klass = globals()[f'{model_name}_{data_name}']
    if klass is None:
        raise ValueError(f"Class '{model_name}' does not exist.")

    model = klass()
    if model_name != 'ViT':
        model = nn.Sequential(*nn.ModuleList(model.children()))
    model.load_state_dict(state_dict_full)
    # evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader):
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
        return False
    else:
        logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

    return True


def _get_base_dataset_from_poisoned(poisoned_ds):
    """Lấy dataset gốc (CIFAR10/MNIST) nằm bên trong lớp backdoor."""
    if hasattr(poisoned_ds, "cifar"):
        return poisoned_ds.cifar
    if hasattr(poisoned_ds, "data"):
        return poisoned_ds.data
    raise AttributeError("Cannot find base dataset inside poisoned dataset")


def _labels_of_poison_sources(poisoned_ds, label_mapping):
    """Lấy danh sách idx ứng viên (có nhãn gốc nằm trong keys(mapping))."""
    base = _get_base_dataset_from_poisoned(poisoned_ds)
    labels = _get_targets(base)
    src_labels = np.array(list(label_mapping.keys()))
    candidates = np.where(np.isin(labels, src_labels))[0]
    return labels, candidates


@torch.no_grad()
def test_backdoor(
        model_name,
        data_name,
        state_dict_full,
        logger,
        args,
        *,
        poison_rate=1.0,
        batch_size=100,
        num_workers=2,
        seed=42
):
    # --- mapping ---
    label_mapping = parse_mapping(args.label_mapping)

    # --- transform ---
    if data_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif data_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError(f"Backdoor test hiện hỗ trợ MNIST và CIFAR10, chưa hỗ trợ '{data_name}'.")

    # --- chọn dataset backdoor ---
    if data_name == "CIFAR10":
        if args.attack_mode == "pixel":
            poisoned_ds = BackdoorCIFAR10(
                root="./data", train=False, download=True, transform=transform_test,
                poison_rate=poison_rate, trigger_size=args.trigger_size,
                trigger_location=args.trigger_location, trigger_color=tuple(args.trigger_color),
                label_mapping=label_mapping, seed=seed
            )
        elif args.attack_mode == "semantic":
            poisoned_ds = SemanticBackdoorCIFAR10(
                root="./data", train=False, download=True, transform=transform_test,
                poison_rate=poison_rate, stripe_width=args.stripe_width,
                alpha=args.alpha, stripe_orientation=args.stripe_orientation,
                label_mapping=label_mapping, seed=seed
            )
        else:
            raise ValueError("attack_mode is not valid for CIFAR10.")
    else:  # MNIST
        if args.attack_mode == "pixel":
            poisoned_ds = BackdoorMNIST(
                root="./data", train=False, download=True, transform=transform_test,
                poison_rate=poison_rate, trigger_size=args.trigger_size,
                trigger_location=args.trigger_location, trigger_value=args.trigger_value,
                label_mapping=label_mapping, seed=seed
            )
        elif args.attack_mode == "semantic":
            poisoned_ds = SemanticBackdoorMNIST(
                root="./data", train=False, download=True, transform=transform_test,
                poison_rate=poison_rate, stripe_width=args.stripe_width,
                alpha=args.alpha, stripe_orientation=args.stripe_orientation,
                label_mapping=label_mapping, seed=seed
            )
        else:
            raise ValueError("attack_mode is not valid for MNIST.")

    # --- lấy nhãn gốc để lọc đúng "bị backdoor & bị lật" ---
    base_ds = _get_base_dataset_from_poisoned(poisoned_ds)
    base_labels = _get_targets(base_ds)  # numpy array nhãn gốc
    poisoned_set = poisoned_ds.poisoned_set  # chỉ số các mẫu đã bị gài

    # tùy chọn: chỉ test một số src-label
    src_filter = getattr(args, "src_filter", None)
    if src_filter is None:
        src_allowed = set(label_mapping.keys())
    else:
        if isinstance(src_filter, (list, tuple, set)):
            src_allowed = set(int(x) for x in src_filter)
        else:
            src_allowed = {int(src_filter)}

    # chọn đúng những index: (i) bị poison, (ii) nhãn gốc trong src_allowed,
    # và (iii) nhãn đã thực sự bị đổi (mapping[src] != src)
    poisoned_and_flipped_indices = [
        i for i in sorted(poisoned_set)
        if int(base_labels[i]) in src_allowed
        and int(label_mapping.get(int(base_labels[i]), int(base_labels[i]))) != int(base_labels[i])
    ]

    # nếu rỗng -> cảnh báo
    if len(poisoned_and_flipped_indices) == 0:
        msg = "[BackdoorTest] Không có mẫu nào thỏa: bị backdoor & thuộc src_filter & bị lật nhãn."
        print(msg)
        logger.log_info(msg)
        return {
            "mapping": label_mapping,
            "poisoned_and_flipped": 0,
            "asr_percent": 0.0,
            "pred_dist": {}
        }

    # Subset chỉ gồm các mẫu cần test
    subset = Subset(WithIndex(poisoned_ds), poisoned_and_flipped_indices)
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Model ---
    if 'MNIST' in data_name:
        klass = globals()[f'{model_name}_MNIST']
    else:
        klass = globals()[f'{model_name}_{data_name}']
    if klass is None:
        raise ValueError(f"Class '{model_name}' does not exist.")

    model = klass()
    if model_name != 'ViT':
        model = nn.Sequential(*nn.ModuleList(model.children()))
    model.load_state_dict(state_dict_full)
    model.eval()

    # --- Đánh giá: accuracy trên tập con == ASR ---
    correct = 0
    total = 0
    pred_counter = Counter()
    src_count = Counter(int(base_labels[i]) for i in poisoned_and_flipped_indices)

    for data, target, idxs in tqdm(test_loader, desc="Eval(backdoor-only)"):
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
        pred_counter.update(pred.tolist())

    asr = 100.0 * correct / total

    msg = (
        f"[BackdoorTest] (ONLY poisoned & flipped) ASR: {correct}/{total} = {asr:.2f}% | "
        f"src_label_counts={dict(src_count)} | top_pred={pred_counter.most_common(5)}"
    )
    print(msg)
    logger.log_info(msg)

    return {
        "mapping": label_mapping,
        "poisoned_and_flipped": int(total),
        "asr_percent": float(asr),
        "src_label_counts": dict(src_count),
        "pred_dist": dict(pred_counter)
    }
