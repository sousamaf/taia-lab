"""
Pipeline mínimo (Aula 1) — simples, mas verdadeiro

O que este pipeline faz:
1) Gera dados sintéticos (classificação binária)
2) Separa treino/validação
3) Treina um MLP pequeno em PyTorch
4) Calcula métricas (loss e accuracy)
5) Salva:
   - pesos do modelo em `models/`
   - relatório de execução em `reports/`

Observação:
- Usamos dados sintéticos para reduzir fricção na Aula 1.
- A partir da Aula 2, este pipeline pode evoluir para registrar experimentos (MLflow),
  ler configs (YAML) e trabalhar com dados reais.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

print('Pipeline mínimo TAIA Aula 1')

# -------------------------
# Configuração mínima
# -------------------------
@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_samples: int = 1200
    n_features: int = 20
    n_classes: int = 2
    test_size: float = 0.2

    hidden_dim: int = 64
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3


# -------------------------
# Utilitários
# -------------------------
def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reprodutibilidade (ok para ensino)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def project_root() -> Path:
    # .../src/taia_lab/pipelines/minimal_pipeline.py -> raiz é 4 níveis acima
    return Path(__file__).resolve().parents[3]


def ensure_dirs() -> Tuple[Path, Path]:
    root = project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir


# -------------------------
# Dados
# -------------------------
def make_data(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=int(cfg.n_features * 0.6),
        n_redundant=int(cfg.n_features * 0.2),
        n_classes=cfg.n_classes,
        class_sep=1.0,
        random_state=cfg.seed,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    return X_train_t, y_train_t, X_val_t, y_val_t


# -------------------------
# Modelo
# -------------------------
class TinyMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Métricas
# -------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_true).sum().item()
    return correct / y_true.numel()


# -------------------------
# Pipeline
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    accs = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
        accs.append(accuracy(logits, yb))

    return (float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(accs)) if accs else float("nan"))


def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val = make_data(cfg)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = TinyMLP(cfg.n_features, cfg.hidden_dim, cfg.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
        val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
        history.append((epoch, train_loss, val_loss, val_acc))
        print(f"epoch={epoch}/{cfg.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Salvar artefatos
    models_dir, reports_dir = ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_path = models_dir / f"tinymlp_aula01_{ts}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
        },
        model_path,
    )

    report_path = reports_dir / f"report_aula01_{ts}.txt"
    lines = []
    lines.append("TAIA — Aula 1 — Pipeline mínimo (verdadeiro)\n")
    lines.append(f"device={device}\n")
    lines.append(f"model={model.__class__.__name__}\n")
    for k, v in asdict(cfg).items():
        lines.append(f"{k}={v}\n")
    lines.append("\nEPOCH,TRAIN_LOSS,VAL_LOSS,VAL_ACC\n")
    for (epoch, train_loss, val_loss, val_acc) in history:
        lines.append(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.6f}\n")
    lines.append(f"\nmodel_path={model_path}\n")

    report_path.write_text("".join(lines), encoding="utf-8")

    print(f"\nSaved model:   {model_path}")
    print(f"Saved report:  {report_path}")


if __name__ == "__main__":
    main()