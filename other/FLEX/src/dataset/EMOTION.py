import torch
import random
from collections import defaultdict

class EMOTIONDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_train_EMOTION(dataset=None, distribution=None):
    random.seed(1)

    train_data = dataset['train']

    train_target_counts = {k: v for k, v in enumerate(distribution)}

    train_by_class = defaultdict(list)
    for text, label in zip(train_data['text'], train_data['label']):
        train_by_class[label].append((text, label))

    train_texts, train_labels = [], []
    for label, count in train_target_counts.items():
        samples = random.sample(train_by_class[label], count)
        train_texts.extend([t for t, _ in samples])
        train_labels.extend([l for _, l in samples])

    print("Train samples:", len(train_texts), {l: train_labels.count(l) for l in set(train_labels)})

    return train_texts, train_labels

def load_test_EMOTION(test_total=1000, dataset=None):
    random.seed(1)
    test_data = dataset['test']
    class_distribution = {
        0: 0.25,
        1: 0.25,
        2: 0.25,
        3: 0.25
    }
    test_target_counts = {k: int(v * test_total) for k, v in class_distribution.items()}
    test_by_class = defaultdict(list)
    for text, label in zip(test_data['text'], test_data['label']):
        test_by_class[label].append((text, label))

    test_texts, test_labels = [], []
    for label, count in test_target_counts.items():
        samples = random.sample(test_by_class[label], count)
        test_texts.extend([t for t, _ in samples])
        test_labels.extend([l for _, l in samples])

    print("Test samples:", len(test_texts), {l: test_labels.count(l) for l in set(test_labels)})

    return test_texts, test_labels