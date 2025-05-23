import torch


class Metric:
    def __init__(self):
        self.value = 0
        self.steps = 0

    def reset(self):
        self.value = 0
        self.steps = 0

    def update(self, value):
        self.steps += 1
        self.value += (value - self.value) / self.steps
        return self.value


def save_checkpoint(
    checkpoint_dir,
    hubert,
    optimizer,
    scaler,
    step,
    loss,
    best,
    logger,
):
    state = {"hubert": hubert.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(),
             "step": step,
             "loss": loss,
             }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model-{step}.pt"
    torch.save(state, checkpoint_path)
    if best:
        best_path = checkpoint_dir / "model-best.pt"
        torch.save(state, best_path)
    logger.info(f"Saved checkpoint: {checkpoint_path.stem}")


def load_checkpoint(
    load_path,
    hubert,
    optimizer,
    scaler,
    rank,
    logger,
):
    logger.info(f"Loading checkpoint from {load_path}")
    checkpoint = torch.load(load_path, map_location={"cuda:0": f"cuda:{rank}"})
    hubert.load_state_dict(checkpoint["hubert"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step, loss = checkpoint.get(
        "step", 0), checkpoint.get("loss", float("inf"))
    return step, loss
