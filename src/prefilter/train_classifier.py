"""
Fine-tune the configured passage classifier for clean vs poisoned passages.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class PassageDataset(Dataset):
    def __init__(self, passages: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            passages,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _extract_passage_labels(
    triplets: list,
    config: dict,
) -> tuple[list[str], list[int]]:
    separator = config["dataset"]["context_separator"]
    passages, labels = [], []

    for t in triplets:
        ctx = t.poisoned_context if hasattr(t, "poisoned_context") else t.context
        split_passages = [p.strip() for p in ctx.split(separator) if p.strip()]
        poisoned_indices = set(getattr(t, "poisoned_passage_indices", []))
        is_poisoned = getattr(t, "is_poisoned", False)

        for i, p in enumerate(split_passages):
            label = int(is_poisoned and i in poisoned_indices)
            passages.append(p)
            labels.append(label)

    return passages, labels


def train_classifier(
    train_triplets: list,
    val_triplets: list,
    config: dict,
) -> None:
    """Train the classifier and save the best validation checkpoint."""
    clf_cfg = config["prefilter"]["classifier"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training {clf_cfg['model']} classifier on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(clf_cfg["model"])
    model = AutoModelForSequenceClassification.from_pretrained(clf_cfg["model"], num_labels=2).to(
        device
    )

    train_passages, train_labels = _extract_passage_labels(train_triplets, config)
    val_passages, val_labels = _extract_passage_labels(val_triplets, config)

    print(f"Train: {len(train_passages)} passages ({sum(train_labels)} poisoned)")
    print(f"Val:   {len(val_passages)} passages ({sum(val_labels)} poisoned)")

    train_ds = PassageDataset(train_passages, train_labels, tokenizer, clf_cfg["max_length"])
    val_ds = PassageDataset(val_passages, val_labels, tokenizer, clf_cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=clf_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=clf_cfg["batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=clf_cfg["lr"])
    total_steps = len(train_loader) * clf_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_f1 = 0.0
    patience = clf_cfg.get("early_stopping_patience", clf_cfg["epochs"])
    epochs_no_improve = 0
    checkpoint_dir = Path(clf_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(clf_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{clf_cfg['epochs']} [train]", unit="batch"
        ) as pbar:
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation pass on held-out triplets.
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{clf_cfg['epochs']} [val]", unit="batch"
            ):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{clf_cfg['epochs']} - loss: {avg_loss:.4f}, val_f1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            print(f"  Saved best checkpoint (f1={best_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best val F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_dir}")
    return best_f1
