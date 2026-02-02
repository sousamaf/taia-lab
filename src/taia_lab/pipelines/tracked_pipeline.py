from __future__ import annotations

"""
Pipeline rastreável (Aula 02) — versionamento + experimento + métrica (MLflow local)

O que este pipeline faz:
1) Gera dados sintéticos (classificação binária)
2) Treina um MLP pequeno em PyTorch
3) Calcula métricas (loss e accuracy)
4) Salva artefatos:
   - pesos do modelo em `models/`
   - relatório em `reports/`
5) Registra o experimento no MLflow (local):
   - parâmetros (config)
   - métricas por época
   - artefatos (modelo + relatório)

Como ver os resultados:
- Execute o pipeline
- Depois rode: `mlflow ui --backend-store-uri ./mlruns`
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import torch
import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Configuração
# -------------------------
@dataclass(frozen=True)
class Config:
    # Dados
    seed: int = 42
    n_samples: int = 1200
    n_features: int = 20
    n_classes: int = 2
    test_size: float = 0.2

    # Modelo/treino
    hidden_dim: int = 64
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3

    # MLflow
    experiment_name: str = "taia-aula02"
    tag_aula: str = "aula02"


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
    # .../src/taia_lab/pipelines/tracked_pipeline.py -> raiz é 4 níveis acima
    return Path(__file__).resolve().parents[3]


def ensure_dirs() -> Tuple[Path, Path, Path]:
    root = project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"
    mlruns_dir = root / "mlruns"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir, mlruns_dir


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
# Treino/avaliação
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

    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(accs)) if accs else float("nan"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAIA Aula 02 — pipeline rastreável (MLflow local)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--experiment-name", type=str, default="taia-aula02")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        experiment_name=args.experiment_name,
    )

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir, reports_dir, mlruns_dir = ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.tag_aula}_{ts}"

    # MLflow local (armazenamento em ./mlruns)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(cfg.experiment_name)

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

    with mlflow.start_run(run_name=run_name):
        # Parâmetros e tags (rastreabilidade)
        mlflow.set_tag("aula", cfg.tag_aula)
        mlflow.set_tag("device", str(device))
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "n_samples": cfg.n_samples,
                "n_features": cfg.n_features,
                "test_size": cfg.test_size,
                "hidden_dim": cfg.hidden_dim,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "model_class": model.__class__.__name__,
            }
        )

        history = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
            val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)

            history.append((epoch, train_loss, val_loss, val_acc))

            # Métricas no MLflow (por época)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(
                f"epoch={epoch}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # Salvar artefatos locais
        model_path = models_dir / f"tinymlp_{cfg.tag_aula}_{ts}.pt"
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, model_path)

        report_path = reports_dir / f"report_{cfg.tag_aula}_{ts}.txt"
        lines = []
        lines.append("TAIA — Aula 02 — Pipeline rastreável (MLflow local)\n")
        lines.append(f"run_name={run_name}\n")
        lines.append(f"experiment={cfg.experiment_name}\n")
        lines.append(f"device={device}\n")
        for k, v in asdict(cfg).items():
            lines.append(f"{k}={v}\n")
        lines.append("\nEPOCH,TRAIN_LOSS,VAL_LOSS,VAL_ACC\n")
        for (e, tr, vl, va) in history:
            lines.append(f"{e},{tr:.6f},{vl:.6f},{va:.6f}\n")
        lines.append(f"\nmodel_path={model_path}\n")
        report_path.write_text("".join(lines), encoding="utf-8")

        # Registrar artefatos no MLflow
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))

        print(f"\nSaved model:  {model_path}")
        print(f"Saved report: {report_path}")
        print("\nPara visualizar no MLflow:")
        print("  mlflow ui --backend-store-uri ./mlruns")
        print("  e abra http://127.0.0.1:5000")


if __name__ == "__main__":
    main()