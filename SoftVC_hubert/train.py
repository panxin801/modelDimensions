import argparse
import torch
import logging
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from tqdm import trange

from hubert.model import (Hubert,)
from hubert.dataset import AcousticUnitsDataset
from hubert.utils import (load_checkpoint, Metric, save_checkpoint)
from hubconf import URLS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

########################################################################################
# Define hyperparameters for training:
########################################################################################

BATCH_SIZE = 8  # 32
LEARNING_RATE = 2e-5
BETAS = (0.9, 0.98)
EPS = 1e-06
WEIGHT_DECAY = 1e-2
MAX_NORM = 10
STEPS = 25000
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 10  # 1000
CHECKPOINT_INTERVAL = 10  # 1000
BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54321"
SEED = 1046


def train(rank, world_size, args):
    dist.init_process_group(
        BACKEND,
        init_method=INIT_METHOD,
        world_size=world_size,
        rank=rank
    )

    torch.manual_seed(SEED)
    torch.cuda.set_device(rank)

    ####################################################################################
    # Setup logging utilities:
    ####################################################################################

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            log_dir / f"{args.checkpoint_dir.stem}.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.setLevel(logging.ERROR)

    writer = SummaryWriter(log_dir) if rank == 0 else None

    ####################################################################################
    # Initialize models
    ####################################################################################

    hubert = Hubert(mask=args.mask).cuda(rank)

    if args.warmstart:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["hubert-discrete"], map_location={"cuda:0": f"cuda:{rank}"}
        )
        consume_prefix_in_state_dict_if_present(
            checkpoint["hubert"], "module.")

        # don't use warmstart weights for label embeddings and proj layer
        del checkpoint["hubert"]["label_embedding.weight"]
        del checkpoint["hubert"]["proj.weight"]
        del checkpoint["hubert"]["proj.bias"]

        hubert.load_state_dict(checkpoint["hubert"], strict=False)

    hubert = DDP(hubert, device_ids=[rank])

    ####################################################################################
    # Initialze optimizer and grad scaler
    ####################################################################################

    optimizer = optim.AdamW(hubert.parameters(),
                            lr=LEARNING_RATE,
                            betas=BETAS,
                            eps=EPS,
                            weight_decay=WEIGHT_DECAY,
                            )
    scaler = amp.GradScaler()

    ####################################################################################
    # Initialize datasets and dataloaders
    ####################################################################################

    train_dataset = AcousticUnitsDataset(
        root=args.dataset_dir,
        train=True
    )
    train_sampler = data.DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True)
    train_loader = data.DataLoader(train_dataset,
                                   collate_fn=train_dataset.collate,
                                   batch_size=BATCH_SIZE,
                                   sampler=train_sampler,
                                   num_workers=8,
                                   pin_memory=True,
                                   shuffle=False,
                                   drop_last=True)

    validation_dataset = AcousticUnitsDataset(
        root=args.dataset_dir,
        train=False
    )
    validation_loader = data.DataLoader(validation_dataset,
                                        batch_size=1,
                                        num_workers=8,
                                        shuffle=False,
                                        pin_memory=True)

    ####################################################################################
    # Load checkpoint if args.resume is set
    ####################################################################################

    if not args.resume is None:
        global_step, best_loss = load_checkpoint(
            load_path=args.resume,
            hubert=hubert,
            optimizer=optimizer,
            scaler=scaler,
            rank=rank,
            logger=logger,
        )
    else:
        global_step, best_loss = 0, float("inf")

    # =================================================================================#
    # Start training loop
    # =================================================================================#

    n_epochs = STEPS // len(train_loader) + 1
    start_epoch = global_step // len(train_loader) + 1

    logger.info("**" * 40)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    logger.info(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
    logger.info(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    logger.info(f"# of GPUS: {torch.cuda.device_count()}")
    logger.info(f"batch size: {BATCH_SIZE}")
    logger.info(f"iterations per epoch: {len(train_loader)}")
    logger.info(f"# of epochs: {n_epochs}")
    logger.info(f"started at epoch: {start_epoch}")
    logger.info("**" * 40 + "\n")

    if args.mask:
        average_masked_loss = Metric()
        average_unmasked_loss = Metric()
        average_masked_accuracy = Metric()
        average_unmasked_accuracy = Metric()

        epoch_masked_loss = Metric()
        epoch_unmasked_loss = Metric()
        epoch_masked_accuracy = Metric()
        epoch_unmasked_accuracy = Metric()
    else:
        average_loss = Metric()
        average_accuracy = Metric()

        epoch_loss = Metric()
        epoch_accuracy = Metric()

    validation_loss = Metric()
    validation_accuracy = Metric()

    for epoch in range(start_epoch, n_epochs + 1):
        train_sampler.set_epoch(epoch)

        hubert.train()
        if args.mask:
            epoch_masked_loss.reset()
            epoch_unmasked_loss.reset()
            epoch_masked_accuracy.reset()
            epoch_unmasked_accuracy.reset()
        else:
            epoch_loss.reset()
            epoch_accuracy.reset()

        # 增加进度条
        pbar = trange(len(train_loader),
                      desc=f"Epoch {epoch}/{n_epochs}",
                      disable=rank != 0)
        for wavs, codes in train_loader:
            pbar.update(1)
            global_step += 1
            wavs, codes = wavs.cuda(rank), codes.cuda(rank)

            ############################################################################
            # Compute training loss
            ############################################################################

            optimizer.zero_grad()

            with amp.autocast():
                logits, mask = hubert(wavs)  # logits=[B,F,D] D=100, mask=[B,F]
                lengths = min(
                    mask.size(-1) if args.mask else float("inf"), codes.size(-1))
                logits = logits[:, :lengths, :]
                codes = codes[:, :lengths]
                if args.mask:
                    mask = mask[:, :lengths]

                    masked_loss = F.cross_entropy(
                        logits[mask], codes[mask])
                    unmasked_loss = F.cross_entropy(
                        logits[~mask], codes[~mask])  # ~mask 取反
                    loss = args.alpha * masked_loss + \
                        (1 - args.alpha) * unmasked_loss  # loss is scalar
                else:
                    loss = F.cross_entropy(logits.transpose(1, 2), codes)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(hubert.parameters(), MAX_NORM)

            scaler.step(optimizer)
            scaler.update()

            if args.mask:
                masked_accuracy = logits[mask].argmax(
                    dim=-1) == codes[mask]
                masked_accuracy = torch.mean(masked_accuracy.float())

                unmasked_accuracy = logits[~mask].argmax(
                    dim=-1) == codes[~mask]
                unmasked_accuracy = torch.mean(unmasked_accuracy.float())
            else:
                accuracy = logits.argmax(dim=-1) == codes
                accuracy = torch.mean(accuracy.float())

            ############################################################################
            # Update and log training metrics
            ############################################################################

            if args.mask:
                average_masked_loss.update(masked_loss.item())
                average_unmasked_loss.update(unmasked_loss.item())
                average_masked_accuracy.update(masked_accuracy.item())
                average_unmasked_accuracy.update(unmasked_accuracy.item())

                epoch_masked_loss.update(masked_loss.item())
                epoch_unmasked_loss.update(unmasked_loss.item())
                epoch_masked_accuracy.update(masked_accuracy.item())
                epoch_unmasked_accuracy.update(unmasked_accuracy.item())
            else:
                average_loss.update(loss.item())
                average_accuracy.update(accuracy.item())

                epoch_loss.update(loss.item())
                epoch_accuracy.update(accuracy.item())

            if rank == 0 and global_step % LOG_INTERVAL == 0:
                if args.mask:
                    writer.add_scalar(
                        "train/masked_loss",
                        average_masked_loss.value,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/unmasked_loss",
                        average_unmasked_loss.value,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/masked_accuracy",
                        average_masked_accuracy.value * 100,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/unmasked_accuracy",
                        average_unmasked_accuracy.value * 100,
                        global_step,
                    )
                    average_masked_loss.reset()
                    average_unmasked_loss.reset()
                    average_masked_accuracy.reset()
                    average_unmasked_accuracy.reset()
                else:
                    writer.add_scalar(
                        "train/loss",
                        average_loss.value,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/accuracy",
                        average_accuracy.value,
                        global_step,
                    )
                    average_loss.reset()
                    average_accuracy.reset()

            # --------------------------------------------------------------------------#
            # Start validation loop
            # --------------------------------------------------------------------------#

            if global_step % VALIDATION_INTERVAL == 0:
                hubert.eval()
                validation_loss.reset()
                validation_accuracy.reset()
                for wavs, codes in validation_loader:
                    wavs, codes = wavs.cuda(rank), codes.cuda(rank)

                    with torch.no_grad():
                        logits, _ = hubert(wavs)
                        logits = logits.transpose(1, 2)

                    loss = F.cross_entropy(logits, codes)

                    accuracy = logits.argmax(dim=1) == codes
                    accuracy = torch.mean(accuracy.float())

                    ####################################################################
                    # Update validation metrics
                    ####################################################################

                    validation_loss.update(loss.item())
                    validation_accuracy.update(accuracy.item())

                hubert.train()

                ############################################################################
                # Log validation metrics
                ############################################################################

                if rank == 0:
                    writer.add_scalar(
                        "validation/unit_loss",
                        validation_loss.value,
                        global_step,
                    )
                    writer.add_scalar(
                        "validation/unit_accuracy",
                        validation_accuracy.value * 100,
                        global_step,
                    )
                    logger.info(
                        f"valid -- epoch: {epoch}, loss: {validation_loss.value:.4f}, accuracy: {validation_accuracy.value * 100:.2f}"
                    )

                ############################################################################
                # Save model checkpoint
                ############################################################################

                new_best = best_loss > validation_loss.value
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        logger.info("-------- new best model found!")
                        best_loss = validation_loss.value

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir=args.checkpoint_dir,
                            hubert=hubert,
                            optimizer=optimizer,
                            scaler=scaler,
                            step=global_step,
                            loss=validation_loss.value,
                            best=new_best,
                            logger=logger,
                        )

            # -----------------------------------------------------------------------------#
            # End validation loop
            # -----------------------------------------------------------------------------#

        ####################################################################################
        # Log training metrics
        ####################################################################################

        logger.info(
            f"""
            train -- epoch: {epoch}, masked loss: {epoch_masked_loss.value:.4f}, unmasked loss: {epoch_unmasked_loss.value:.4f}, 
                     masked accuracy: {epoch_masked_accuracy.value * 100:.2f}, umasked accuracy: {epoch_unmasked_accuracy.value * 100:.2f}
            """
        )

        # ==================================================================================#
        # End training loop
        # ==================================================================================#

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HuBERT soft content encoder.")
    parser.add_argument("dataset_dir",
                        metavar="dataset-dir",
                        help="path to the data directory.",
                        type=Path,
                        )
    parser.add_argument(
        "checkpoint_dir",
        metavar="checkpoint-dir",
        help="path to the checkpoint directory.",
        type=Path,
    )
    parser.add_argument(
        "--resume",
        help="path to the checkpoint to resume from.",
        type=Path,
    )
    parser.add_argument(
        "--warmstart",
        help="whether to initialize from the fairseq HuBERT checkpoint.",
        action="store_true",
    )
    parser.add_argument(
        "--mask",
        help="whether to use input masking.",
        action="store_true",
    )
    parser.add_argument(
        "--alpha",
        help="weight for the masked loss.",
        default=1,
        type=float,
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    # mp.spawn(
    #     train,
    #     args=(world_size, args),
    #     nprocs=world_size,
    #     join=True
    # )
    train(0, 1, args)
