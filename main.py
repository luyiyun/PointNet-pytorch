import os

import torch
import torch.optim as optim
import torch.utils.data as data

from dataset import (
    FileDataset, SampleTransfer, CircleNormTransfer,
    RandomJitterTransfer, RandomRotationTransfer
)
from models import PointNet, CEwithReg
from train import train, evaluate
import utils


def main():
    # 读取配置
    config = utils.load_json("./config.json")
    data_conf, model_conf, train_conf = (
        config["data"], config["model"], config["train"]
    )
    device = torch.device(train_conf["device"])
    task_name = train_conf["name"] if train_conf["name"] is not None\
        else utils.task_name_generate()

    # 读取数据集
    train_trans = [
        SampleTransfer(data_conf["npoints"], data_conf["sample_method"]),
        CircleNormTransfer(),
        RandomRotationTransfer(),
        RandomJitterTransfer(data_conf["jitter_std"], data_conf["jitter_clip"])
    ]
    eval_trans = [
        SampleTransfer(data_conf["npoints"], data_conf["sample_method"]),
        CircleNormTransfer(),
    ]
    train_dat = FileDataset.ModelNet(config["data"]["dir"], phase="train")
    test_dat = FileDataset.ModelNet(
        config["data"]["dir"], phase="test",
        label_encoder=train_dat.kwargs["label_encoder"]
    )
    train_dat, eval_dat = train_dat.split(0.1, True, config["seed"], True)
    train_dat.set_transfers(*train_trans)
    eval_dat.set_transfers(*eval_trans)
    test_dat.set_transfers(*eval_trans)

    loaders = {
        "train": data.DataLoader(
            train_dat, train_conf["batch_size"], True,
            num_workers=train_conf["njobs"]
        ),
        "eval": data.DataLoader(
            eval_dat, train_conf["batch_size"], False,
            num_workers=train_conf["njobs"]
        ),
        "test": data.DataLoader(
            test_dat, train_conf["batch_size"], False,
            num_workers=train_conf["njobs"]
        )
    }
    # 构建模型
    net = PointNet(
        train_dat.channels, train_dat.nlabels,
        **model_conf["pointnet"], stn3_kwargs=model_conf["stn3"],
        stnk_kwargs=model_conf["stnk"]
    ).to(device)
    criterion = CEwithReg(model_conf["reg_w"])
    optimizer = optim.Adam(net.parameters(), lr=train_conf["lr"])

    # 训练
    net, hist, best = train(
        net, criterion, optimizer, loaders, train_conf["epoch"],
        device, task_name
    )
    test_loss, test_acc = evaluate(net, criterion, loaders["test"], device)
    best["test_loss"] = test_loss
    best["test_acc"] = test_acc

    # 保存结果
    task_dir = os.path.join("RESULTS", task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    torch.save(net, os.path.join(task_dir, "model.pth"))
    utils.dump_json(best, os.path.join(task_dir, "best.json"))
    utils.dump_json(hist, os.path.join(task_dir, "hist.json"))
    utils.dump_json(config, os.path.join(task_dir, "config.json"))


if __name__ == "__main__":
    main()
