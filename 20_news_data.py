import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    categories = ["alt.atheism", "soc.religion.christian"]
    random_state: int = 42
    val_ratio: float = 0.2
    min_freq: int = 3
    max_len: int = 400
    embed_dim: int = 128
    hidden_dim: int = 160
    dropout: float = 0.4
    batch_size: int = 32
    lr: float = 8e-4
    weight_decay: float = 1e-4
    epochs: int = 18
    patience: int = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return text.split()


def build_vocab(texts, min_freq: int = 2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    stoi = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            stoi[token] = len(stoi)
    return stoi


def encode_text(text: str, stoi, max_len: int):
    ids = [stoi.get(tok, stoi["<UNK>"]) for tok in tokenize(text)]
    if len(ids) == 0:
        ids = [stoi["<UNK>"]]
    return ids[:max_len]


class NewsDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len: int):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sequences = [encode_text(t, stoi, max_len) for t in texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, lengths, labels


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, lengths):
        emb = self.embedding(inputs)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        feat = self.dropout(hidden[-1])
        logits = self.fc(feat).squeeze(1)
        return logits


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(inputs, lengths)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, lengths, labels in dataloader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def load_data_and_split(config: Config):
    workspace_root = Path(__file__).resolve().parent
    local_root = workspace_root / "20news-bydate"
    local_train = local_root / "20news-bydate-train"
    local_test = local_root / "20news-bydate-test"

    if local_train.exists() and local_test.exists():
        print(f"检测到本地数据目录: {local_root}")

        label_to_id = {cat: i for i, cat in enumerate(config.categories)}

        def load_local_split(split_dir: Path):
            texts = []
            labels = []
            for cat in config.categories:
                cat_dir = split_dir / cat
                if not cat_dir.exists():
                    raise FileNotFoundError(f"未找到类别目录: {cat_dir}")
                for file_path in sorted(cat_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    text = file_path.read_text(encoding="latin1", errors="ignore")
                    texts.append(preprocess_text(text))
                    labels.append(label_to_id[cat])
            return texts, np.asarray(labels, dtype=np.int64)

        train_texts, train_labels = load_local_split(local_train)
        test_texts, test_labels = load_local_split(local_test)
    else:
        print("未检测到本地目录，回退到 fetch_20newsgroups 在线加载。")
        newsgroups_train = fetch_20newsgroups(
            subset="train",
            categories=config.categories,
            remove=("headers", "footers", "quotes"),
        )
        newsgroups_test = fetch_20newsgroups(
            subset="test",
            categories=config.categories,
            remove=("headers", "footers", "quotes"),
        )

        train_texts = [preprocess_text(doc) for doc in newsgroups_train.data]
        test_texts = [preprocess_text(doc) for doc in newsgroups_test.data]
        train_labels = np.asarray(newsgroups_train.target, dtype=np.int64)
        test_labels = np.asarray(newsgroups_test.target, dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        train_texts,
        train_labels,
        test_size=config.val_ratio,
        random_state=config.random_state,
        stratify=train_labels,
    )

    stoi = build_vocab(x_train, min_freq=config.min_freq)

    return x_train, x_val, test_texts, y_train, y_val, test_labels, stoi


def main():
    config = Config()
    set_seed(config.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    x_train, x_val, x_test, y_train, y_val, y_test, stoi = load_data_and_split(config)
    print(f"词汇表大小: {len(stoi)}")
    print(f"原始训练集大小(newsgroups_train): {len(x_train) + len(x_val)}")
    print(f"拆分后训练集: {len(x_train)}, 验证集: {len(x_val)}")
    print(f"测试集大小(newsgroups_test): {len(x_test)}")

    train_ds = NewsDataset(x_train, y_train, stoi, config.max_len)
    val_ds = NewsDataset(x_val, y_val, stoi, config.max_len)
    test_ds = NewsDataset(x_test, y_test, stoi, config.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = GRUClassifier(
        vocab_size=len(stoi),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.patience:
            print("验证集指标连续未提升，触发早停。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"测试集结果: loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    main()