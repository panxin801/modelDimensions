# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

import warnings
import argparse
import logging
import torch
import platform
from collections import OrderedDict
from pathlib import Path
from pytorch_lightning import (Trainer, seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from process_ckpt import my_save
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from AR.utils import get_newest_ckpt


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest  # False
        self.if_save_every_weights = if_save_every_weights  # True
        self.half_weights_save_dir = half_weights_save_dir  # GPT_weights_v4
        self.exp_name = exp_name  # TestA1
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):  # 如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove(f"{self.dirpath}/{name}")
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = f"GPT-e{trainer.current_epoch + 1}"
                    # torch.save(
                    # print(os.environ)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            f"{self.half_weights_save_dir}/{self.exp_name}-e{trainer.current_epoch + 1}.ckpt",
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config)  # config is a dict

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(  # 检查点回调函数
        config=config,
        if_save_latest=config["train"]["if_save_latest"],  # False
        if_save_every_weights=config["train"]["if_save_every_weights"],  # True
        # GPT_weights_v4
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],  # 1
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(
        name=output_dir.stem, save_dir=output_dir)  # 记录日志
    # MASTER_ADDR和USE_LIBUV用于分布式训练，在windows上需要设置为0
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        limit_val_batches=0,
        devices=-1 if torch.cuda.is_available() else 1,
        benchmark=False,
        fast_dev_run=False,
        strategy=DDPStrategy(
            process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
        if torch.cuda.is_available()
        else "auto",
        precision=config["train"]["precision"],  # 32
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir)  # AR 模型实例

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        # 'logs/TestA1/6-name2semantic.tsv'
        train_semantic_path=config["train_semantic_path"],
        # 'logs/TestA1/2-name2text.txt'
        train_phoneme_path=config["train_phoneme_path"],
        # dev_semantic_path=args.dev_semantic_path,
        # dev_phoneme_path=args.dev_phoneme_path
    )  # AR 数据模块实例，负责数据加载

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print(f"ckpt_path: {ckpt_path}")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)  # 开始训练


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c",
                        type=str,
                        default="TEMP/tmp_s1.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    logging.info(str(args))
    main(args)
