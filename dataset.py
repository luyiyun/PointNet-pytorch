import os

import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def off_reader(fn):
    """ .off --> ndarray """
    with open(fn, "r") as f:
        f.readline()
        num_p, _, _ = f.readline().strip().split(" ")
        num_p = int(num_p)
        content = [
            [float(i) for i in f.readline().strip().split(" ")]
            for _ in range(num_p)
        ]

    return np.array(content).astype("float32")


class SampleTransfer:

    def __init__(self, n_points, method="random"):
        if method not in ["random", "farthest"]:
            raise ValueError(
                "method must be in one of 'random' and 'farthest'."
            )
        self.n_points = n_points
        self.method = method

    def __call__(self, t):
        if self.method == "random":
            return self.random_sample(t)
        elif self.method == "farthest":
            return self.farthest_sample(t)

    def random_sample(self, t):
        inds = np.random.choice(t.shape[0], size=self.n_points)
        return t[inds]

    def farthest_sample(self, t):
        N = t.shape[0]
        centroids = np.zeros((self.n_points,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(self.n_points):
            centroids[i] = farthest
            centroid = t[farthest]
            dist = np.sum((t - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        t = t[centroids.astype("int32")]
        return t


class CircleNormTransfer:

    def __call__(self, t):
        t -= t.mean(axis=0)
        t /= np.max(np.sqrt((t ** 2).sum(-1)))
        return t


class RandomRotationTransfer:

    def __init__(self, axis=2):
        self.rotate_dims = list(range(3))
        self.rotate_dims.pop(axis)

    def __call__(self, t):
        theta = np.random.uniform(0., np.pi * 2)
        rotation_mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        t[:, self.rotate_dims] = np.matmul(
            t[:, self.rotate_dims], rotation_mat
        )
        return t


class RandomJitterTransfer:

    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, t):
        jitter = np.random.normal(0., self.std, size=t.shape)
        jitter = np.clip(jitter, -self.clip, self.clip)
        return t + jitter


class FileDataset(Dataset):

    def __init__(self, fns, labels, reader=off_reader, **kwargs):
        super().__init__()
        self.fns = fns
        self.labels = labels
        self.reader = reader
        self.transfers = None
        self.kwargs = kwargs

    def __getitem__(self, idx):
        sample = self.reader(self.fns[idx])
        if self.transfers is not None:
            for t in self.transfers:
                sample = t(sample)
        return sample.T, self.labels[idx]  # 输出的tensor channel的维度是在第二个的

    def __len__(self):
        return len(self.fns)

    def set_transfers(self, *transfers):
        self.transfers = transfers
        return self

    def split(self, test_size, shuffle=True, random_seed=None, stratify=None):
        if stratify is True:
            stratify_arr = self.labels
        elif not stratify:
            stratify_arr = stratify
        else:
            stratify_arr = None

        train_fns, test_fns, train_lbs, test_lbs = train_test_split(
            self.fns, self.labels, test_size=test_size,
            random_state=random_seed, shuffle=shuffle, stratify=stratify_arr
        )
        return (
            self.__class__(train_fns, train_lbs, self.reader, **self.kwargs),
            self.__class__(test_fns, test_lbs, self.reader, **self.kwargs)
        )

    @classmethod
    def ModelNet(cls, path, phase="train", label_encoder=None):
        classes, class_dirs = [], []
        for d in os.listdir(path):
            abs_d = os.path.join(path, d)
            if os.path.isdir(abs_d):
                classes.append(d)
                class_dirs.append(abs_d)

        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(classes)
        else:
            label_encoder = label_encoder
        classes = label_encoder.transform(classes)

        fns, labels = [], []
        for class_dir, label in zip(class_dirs, classes):
            class_dir2 = os.path.join(class_dir, phase)
            for fn in os.listdir(class_dir2):
                if fn.endswith(".off"):
                    fns.append(os.path.join(class_dir2, fn))
                    labels.append(label)

        return cls(fns, labels, label_encoder=label_encoder)

    @property
    def channels(self):
        x, _ = self.__getitem__(0)
        return x.shape[0]

    @property
    def nlabels(self):
        return len(self.kwargs["label_encoder"].classes_)


if __name__ == "__main__":
    import utils

    config = utils.load_json("./config.json")
    transfers = [
        SampleTransfer(1024, "farthest"),
        CircleNormTransfer(),
        RandomRotationTransfer(),
        RandomJitterTransfer()
    ]
    train_dat = FileDataset.ModelNet(
        config["data"]["dir"], phase="train"
    ).set_transfers(*transfers)
    test_dat = FileDataset.ModelNet(
        config["data"]["dir"], phase="test",
        label_encoder=train_dat.kwargs["label_encoder"]
    ).set_transfers(*transfers)
    for x, y in train_dat:
        # x, y = train_dat[130]
        print(x.shape)
        print(y.shape)

    # fig, axs = utils.plant_visual(x)
    # fig.savefig("visual.png")
