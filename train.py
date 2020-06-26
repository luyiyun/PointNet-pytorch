from math import inf
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metric import MetricHistroy, Loss, Acc


def train(
    model, criterion, optimizer, dataloaders, epoch, device,
    task_name=None
):
    writer = SummaryWriter(log_dir="runs/%s" % task_name)
    metric_hist = MetricHistroy(writer)
    best = {"epoch": -1, "score": -inf}
    best_model = model.state_dict()

    for e in tqdm(range(epoch)):
        model.train()
        for x, y in dataloaders["train"]:
            x = x.to(device, torch.float)
            y = y.to(device, torch.long)
            pred, _, trans_feat = model(x)
            loss = criterion(pred, y, trans_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_hist.add(loss, pred, y)
        metric_hist.record(e, "train")

        model.eval()
        with torch.no_grad():
            for x, y in dataloaders["eval"]:
                x = x.to(device, torch.float)
                y = y.to(device, torch.long)
                pred, _, trans_feat = model(x)
                loss = criterion(pred, y, trans_feat)

                metric_hist.add(loss, pred, y)
        metric_hist.record(e, "eval")
        writer.flush()

        score = metric_hist["value"][-1]
        if score > best["score"]:
            best["epoch"] = e
            best["score"] = score
            best_model = deepcopy(model.state_dict())

    model.load_state_dict(best_model)
    writer.close()

    return model, metric_hist.history, best


def evaluate(model, criterion, dataloader, device):
    loss_obj, acc_obj = Loss(), Acc()
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, torch.float)
            y = y.to(device, torch.long)
            pred, _, trans_feat = model(x)
            loss = criterion(pred, y, trans_feat)

            loss_obj.add(loss, x.size(0))
            acc_obj.add(pred, y)

    return loss_obj.value(), acc_obj.value()
