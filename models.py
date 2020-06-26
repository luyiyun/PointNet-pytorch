import torch
import torch.nn as nn


class STN(nn.Module):

    def __init__(self, inpc, p2p_hiddens, fc_hiddens, trans_xyz=True):
        super().__init__()
        self.inpc = inpc
        self.trans_xyz = trans_xyz
        if self.trans_xyz:
            assert self.inpc == 3

        p2p_part = []
        for i, o in zip([inpc] + p2p_hiddens[:-1], p2p_hiddens):
            p2p_part.append(nn.Sequential(
                nn.Conv1d(i, o, 1),
                nn.BatchNorm1d(o),
                nn.ReLU()
            ))
        self.p2p_part = nn.Sequential(*p2p_part)

        fc_part = []
        for i, o in zip(p2p_hiddens[-1:] + fc_hiddens[:-1], fc_hiddens):
            fc_part.append(nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.ReLU()
            ))
        fc_part.append(nn.Linear(fc_hiddens[-1], inpc*inpc))
        self.fc_part = nn.Sequential(*fc_part)

    def forward(self, t):
        D = t.size(1)
        x = self.p2p_part(t)
        x = x.max(dim=2)[0]
        x = self.fc_part(x)

        x = x.reshape(-1, self.inpc, self.inpc)
        iden = torch.eye(self.inpc, self.inpc).unsqueeze(0).to(x)
        x += iden  # 加了这一步相当于在这里加了一个residual connection

        if self.trans_xyz and D > 3:
            t, f = t[:, :self.trans_dims, :], t[:, self.trans_dims:, :]
        t = torch.bmm(t.transpose(1, 2), x)
        t = t.transpose(1, 2)
        if self.trans_xyz and D > 3:
            t = torch.cat([t, f], dim=1)

        return t, x


class PointNet(nn.Module):

    def __init__(
        self, inpc, outc, p2p_hiddens, fc_hiddens, stn3_kwargs,
        first_conv=64, stnk_kwargs=None
    ):
        super().__init__()
        self.stn3 = STN(inpc, **stn3_kwargs, trans_xyz=True)
        if stnk_kwargs is not None:
            self.stnk = STN(first_conv, **stnk_kwargs, trans_xyz=False)
        else:
            self.stnk = None
        self.first_conv = nn.Sequential(
            nn.Conv1d(inpc, first_conv, 1),
            nn.BatchNorm1d(first_conv),
            nn.ReLU()
        )
        p2p_net = []
        for i, o in zip([first_conv] + p2p_hiddens[:-1], p2p_hiddens):
            p2p_net.append(nn.Sequential(
                nn.Conv1d(i, o, 1),
                nn.BatchNorm1d(o),
                nn.ReLU()
            ))
        self.p2p_net = nn.Sequential(*p2p_net)
        fc_net = []
        for i, o in zip(p2p_hiddens[-1:] + fc_hiddens[:-1], fc_hiddens):
            fc_net.append(nn.Sequential(
                nn.Linear(i, o),
                nn.BatchNorm1d(o),
                nn.ReLU()
            ))
        fc_net.append(nn.Linear(fc_hiddens[-1], outc))
        self.fc_net = nn.Sequential(*fc_net)

    def forward(self, t):
        x, trans = self.stn3(t)
        x = self.first_conv(x)
        if self.stnk is not None:
            x, trans_feat = self.stnk(x)
        x = self.p2p_net(x)
        x = x.max(dim=2)[0]
        x = self.fc_net(x)
        res = [x, trans]
        if self.stnk is not None:
            res.append(trans_feat)
        return res


class CEwithReg(nn.Module):
    def __init__(self, reg_weight=1.):
        super().__init__()
        self.reg_weight = reg_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target, trans):
        l1 = self.ce(input, target)
        l2 = self.reg(trans)
        return l1 + self.reg_weight * l2

    @staticmethod
    def reg(trans):
        d = trans.size(1)
        iden = torch.eye(d).unsqueeze(0).to(trans)
        loss = (trans.bmm(trans.transpose(1, 2))-iden).norm(dim=(1, 2)).mean()
        return loss


if __name__ == "__main__":
    from utils import load_json

    model_conf = load_json("./config.json")["model"]
    inpt = torch.rand(100, 3, 1024)
    net = PointNet(
        3, 10, **model_conf["pointnet"], stn3_kwargs=model_conf["stn3"],
        stnk_kwargs=model_conf["stnk"]
    )
    output = net(inpt)
    print(output.shape)
