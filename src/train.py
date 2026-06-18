from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config, num_classes
from src.dataset import ISICDataset, get_splits, pick_device, train_transform
from src.model import build_model
from src.utils import dump_json, ensure_dir, run_stamp


def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_training(cfg, name_override=None):
    device = pick_device()
    print("device:", device)

    train_cfg, model_cfg = cfg["train"], cfg["model"]
    transform = train_transform(model_cfg["image_size"])

    train_df, val_df = get_splits(cfg)
    print("train:", len(train_df), "val:", len(val_df))

    pin = device.type == "cuda"
    train_loader = DataLoader(ISICDataset(train_df, cfg["paths"]["images_dir"], transform),
                              batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=train_cfg["num_workers"], pin_memory=pin)
    val_loader = DataLoader(ISICDataset(val_df, cfg["paths"]["images_dir"], transform),
                            batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=train_cfg["num_workers"], pin_memory=pin)

    torch.manual_seed(cfg["seed"])
    model = build_model(model_cfg["backbone"], num_classes(cfg), model_cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    runs_dir = ensure_dir(cfg["paths"]["runs_dir"])
    checkpoint_path = Path(cfg["paths"]["checkpoint"])

    history = []
    best_acc, best_epoch = -1.0, 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print("epoch %d  train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f"
              % (epoch, tr_loss, tr_acc, val_loss, val_acc))
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), checkpoint_path)

    dump_json(runs_dir / "history.json", history)

    run_name = run_stamp(name_override if name_override is not None else cfg.get("name"))
    result = {
        "run_name": run_name,
        "config_path": cfg["config_path"],
        "backbone": model_cfg["backbone"],
        "epochs": train_cfg["epochs"],
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "checkpoint": str(checkpoint_path),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "history": history,
    }
    result_path = dump_json(runs_dir / "result.json", result)
    print("saved checkpoint:", checkpoint_path, "(epoch", best_epoch, "val_acc=%.4f)" % best_acc)
    return result_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config", type=click.Path(path_type=Path), default=None)
@click.option("--name", default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--lr", type=float, default=None)
@click.option("--batch-size", type=int, default=None)
def main(config, name, epochs, lr, batch_size):
    cfg = load_config(config)
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if lr is not None:
        cfg["train"]["lr"] = lr
    if batch_size is not None:
        cfg["train"]["batch_size"] = batch_size
    click.echo(run_training(cfg, name))


if __name__ == "__main__":
    main()
