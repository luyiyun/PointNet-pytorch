class Loss:

    def __init__(self):
        self.all = 0.
        self.count = 0

    def add(self, loss, batch):
        self.all += loss.detach().cpu().item() * batch
        self.count += batch

    def value(self):
        return self.all / self.count


class Acc:

    def __init__(self):
        self.correct = 0
        self.count = 0

    def add(self, pred, target):
        self.count += pred.size(0)
        c = (pred.argmax(dim=1) == target).sum().detach().cpu().item()
        self.correct += c

    def value(self):
        return self.correct / self.count


class MetricHistroy:

    def __init__(self, writer):
        self.history = {
            "epoch": [], "phase": [],
            "metric": [], "value": []
        }
        self.writer = writer
        self.refresh()

    def __getitem__(self, key):
        return self.history[key]

    def add(self, loss, pred, target):
        self.loss_obj.add(loss, pred.size(0))
        self.acc_obj.add(pred, target)

    def record(self, epoch, phase):
        self.history["epoch"].extend([epoch]*2)
        self.history["phase"].extend([phase]*2)
        self.history["metric"].extend(["loss", "acc"])
        self.history["value"].extend([
            self.loss_obj.value(), self.acc_obj.value()
        ])
        self.writer.add_scalar(
            "loss/%s" % phase, self.history["value"][-2], epoch
        )
        self.writer.add_scalar(
            "acc/%s" % phase, self.history["value"][-1], epoch
        )
        self.refresh()

    def refresh(self):
        self.loss_obj = Loss()
        self.acc_obj = Acc()
