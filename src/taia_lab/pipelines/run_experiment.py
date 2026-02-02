from __future__ import annotations

"""
Runner de experimentos (Aula 03)
- Lê YAML em /configs
- Executa treino (TinyMLP em PyTorch)
- Calcula métricas (val_loss, val_acc)
- Salva artefatos em models/ e reports/
- Registra parâmetros/métricas/artefatos no MLflow (local em ./mlruns)

Uso:
  python -m taia_lab.pipelines.run_experiment --config configs/exp01_baseline.yaml

Visualizar no MLflow:
  mlflow ui --backend-store-uri ./mlruns
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import mlflow
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------
# Tipos e Config
# -------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    # Identidade
    name: str
    description: str

    # Dados
    seed: int
    n_samples: int
    n_features: int
    test_size: float

    # Treino
    epochs: int
    batch_size: int
    lr: float

    # Modelo
    hidden_dim: int
    n_classes: int

    # Tracking
    mlflow_experiment_name: str
    tags: Dict[str, str]


# -------------------------
# Utils
# -------------------------
def project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Não foi possível detectar a raiz do projeto (pyproject.toml/.git).")


def ensure_dirs() -> Tuple[Path, Path, Path]:
    root = project_root()
    models_dir = root / "models"
    reports_dir = root / "reports"
    mlruns_dir = root / "mlruns"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, reports_dir, mlruns_dir


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reprodutibilidade (adequado para ensino)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de config não encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML inválido: esperado um dicionário no topo.")
    return data


def parse_config(y: Dict[str, Any]) -> ExperimentConfig:
    # Campos obrigatórios, com falhas explícitas (didático)
    exp = y.get("experiment", {})
    data = y.get("data", {})
    train = y.get("train", {})
    model = y.get("model", {})
    tracking = y.get("tracking", {})

    name = exp.get("name")
    description = exp.get("description", "").strip()

    seed = int(data.get("seed"))
    n_samples = int(data.get("n_samples"))
    n_features = int(data.get("n_features"))
    test_size = float(data.get("test_size"))

    epochs = int(train.get("epochs"))
    batch_size = int(train.get("batch_size"))
    lr = float(train.get("lr"))

    hidden_dim = int(model.get("hidden_dim"))
    n_classes = int(model.get("n_classes", 2))

    tool = tracking.get("tool", "mlflow")
    if tool != "mlflow":
        raise ValueError("Somente tracking.tool=mlflow é suportado nesta versão.")
    mlflow_experiment_name = tracking.get("experiment_name")
    tags = tracking.get("tags", {}) or {}
    tags = {str(k): str(v) for k, v in tags.items()}

    missing = []
    if not name:
        missing.append("experiment.name")
    if not mlflow_experiment_name:
        missing.append("tracking.experiment_name")
    if missing:
        raise ValueError(f"Campos obrigatórios ausentes no YAML: {', '.join(missing)}")

    return ExperimentConfig(
        name=str(name),
        description=str(description),
        seed=seed,
        n_samples=n_samples,
        n_features=n_features,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        n_classes=n_classes,
        mlflow_experiment_name=str(mlflow_experiment_name),
        tags=tags,
    )


# -------------------------
# Dados
# -------------------------
def make_data(cfg: ExperimentConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )


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


@torch.no_grad()
def accuracy(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_true).sum().item()
    return correct / y_true.numel()


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


# -------------------------
# Runner
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAIA Aula 03 — Runner de experimentos via YAML")
    p.add_argument("--config", type=str, required=True, help="Caminho do YAML do experimento (ex: configs/exp01.yaml)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)

    y = load_yaml(cfg_path)
    cfg = parse_config(y)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir, reports_dir, mlruns_dir = ensure_dirs()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.name}_{ts}"

    # MLflow local (em ./mlruns)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(cfg.mlflow_experiment_name)

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
        # Tags e parâmetros (rastreabilidade)
        mlflow.set_tag("config_path", str(cfg_path.relative_to(root)) if cfg_path.is_relative_to(root) else str(cfg_path))
        mlflow.set_tag("device", str(device))
        if cfg.description:
            mlflow.set_tag("description", cfg.description)
        for k, v in cfg.tags.items():
            mlflow.set_tag(k, v)

        mlflow.log_params(
            {
                "seed": cfg.seed,
                "n_samples": cfg.n_samples,
                "n_features": cfg.n_features,
                "test_size": cfg.test_size,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "hidden_dim": cfg.hidden_dim,
                "n_classes": cfg.n_classes,
                "model_class": model.__class__.__name__,
            }
        )

        history = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device)
            val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)
            history.append((epoch, train_loss, val_loss, val_acc))

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            print(
                f"[{cfg.name}] epoch={epoch}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        # Métricas finais (úteis para relatório do aluno)
        final_val_loss, final_val_acc = history[-1][2], history[-1][3]
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_val_acc", final_val_acc)

        # Artefatos locais
        model_path = (models_dir / f"{cfg.name}_{ts}.pt").resolve()

        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
            },
            model_path,
        )

        report_path = (reports_dir / f"report_{cfg.name}_{ts}.txt").resolve()
        lines = []
        lines.append("TAIA — Aula 03 — Experimento via YAML\n\n")
        lines.append(f"experiment_name={cfg.name}\n")
        lines.append(f"description={cfg.description}\n")
        lines.append(f"config_file={cfg_path}\n")
        lines.append(f"mlflow_experiment={cfg.mlflow_experiment_name}\n")
        lines.append(f"run_name={run_name}\n")
        lines.append(f"device={device}\n\n")

        lines.append("PARAMS\n")
        lines.append(f"seed={cfg.seed}\n")
        lines.append(f"n_samples={cfg.n_samples}\n")
        lines.append(f"n_features={cfg.n_features}\n")
        lines.append(f"test_size={cfg.test_size}\n")
        lines.append(f"epochs={cfg.epochs}\n")
        lines.append(f"batch_size={cfg.batch_size}\n")
        lines.append(f"lr={cfg.lr}\n")
        lines.append(f"hidden_dim={cfg.hidden_dim}\n")
        lines.append(f"n_classes={cfg.n_classes}\n\n")

        lines.append("EPOCH,TRAIN_LOSS,VAL_LOSS,VAL_ACC\n")
        for (e, tr, vl, va) in history:
            lines.append(f"{e},{tr:.6f},{vl:.6f},{va:.6f}\n")

        lines.append("\nFINAL\n")
        lines.append(f"final_val_loss={final_val_loss:.6f}\n")
        lines.append(f"final_val_acc={final_val_acc:.6f}\n")
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