import torch

class AGNEWS_DATASET(torch.utils.data.Dataset):
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

from collections import defaultdict
import random

def load_test_AGNEWS(num_samples, dataset):
    test_data = dataset['test']
    
    # AGNEWS có 4 class, chia đều theo num_samples
    distribution = [num_samples // 4] * 4
    test_target_counts = {k: v for k, v in enumerate(distribution)}
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