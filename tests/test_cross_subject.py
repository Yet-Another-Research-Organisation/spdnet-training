"""Cross-subject protocol (spdnet_training.cross_subject) on a fake dataset."""

import copy

import numpy as np
import pytest
import torch
from spdnet_datasets import DatasetManager
from spdnet_datasets.moabb import subject_fold_splits
from yetanotherspdnet.model import SPDnet

from spdnet_training.cross_subject import (
    DomainBatchNorm,
    DomainBatchSampler,
    _bn_layers,
    create_dataloaders,
)

N_CH, N_SUBJECTS, N_PER_SESSION = 4, 4, 12


@DatasetManager.register_dataset("fake_subjects")
class FakeSubjects(torch.utils.data.Dataset):
    """SPD covariances of 2 classes in 2 sessions, each session with its own scale."""

    def __init__(self, subject_folds=3, split=None, seed=0, verbose=False, **_):
        rng = np.random.default_rng(seed)
        n = 2 * N_PER_SESSION
        self.subjects = np.repeat(np.arange(1, N_SUBJECTS + 1), n)
        self.sessions = np.tile(np.repeat(["0", "1"], N_PER_SESSION), N_SUBJECTS)
        self.groups = np.array([f"{s}-{t}" for s, t in zip(self.subjects, self.sessions, strict=True)])
        self.labels = np.tile([0, 1], N_SUBJECTS * N_PER_SESSION)  # cues alternate
        A = rng.standard_normal((len(self.labels), N_CH, 3 * N_CH))
        scale = (1 + self.labels) * self.subjects * (1 + self.sessions.astype(int))
        A *= np.sqrt(scale)[:, None, None]
        self.covs = torch.from_numpy(A @ A.transpose(0, 2, 1) / (3 * N_CH))
        self.classes = ["a", "b"]
        self.samples = [{"class_name": self.classes[k]} for k in self.labels]
        self.splits = subject_fold_splits(self.labels, self.subjects, self.sessions,
                                          subject_folds)
        self.splits["plain"] = {k: self.splits["subject_fold_0"][k]
                                for k in ("train", "val", "test")}

    def get_num_classes(self):
        return 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.covs[i], int(self.labels[i])


CONFIG = {"name": "fake_subjects", "split_strategy": "subject_fold_0", "batch_size": 5,
          "num_workers": 0, "seed": 0, "verbose": False, "subject_folds": 3,
          "calib_sizes": [0, 4, None]}


def test_sampler_batches_hold_one_session_and_cover_the_trials():
    ds = FakeSubjects()
    idx = ds.splits["subject_fold_0"]["train"]
    sampler = DomainBatchSampler(idx, ds.groups, batch_size=5, seed=0)
    first, again = list(sampler), list(sampler)
    sampler.sampler.set_epoch(1)  # as Lightning does at each epoch
    second = list(sampler)
    assert len(first) == len(sampler) and first == again  # set by the epoch only
    assert all(len(set(ds.groups[b])) == 1 for b in first)
    assert sorted(i for b in first for i in b) == sorted(idx.tolist())
    assert min(len(b) for b in first) >= 3 and first != second  # reshuffled


def test_loaders_follow_the_split_and_group_by_session():
    train, val, test, n_classes, cb = create_dataloaders(CONFIG)
    ds, fold = train.dataset, FakeSubjects().splits["subject_fold_0"]
    assert n_classes == 2 and isinstance(cb, DomainBatchNorm)
    for loader, part in ((val, "val"), (test, "test")):
        batches = list(loader.batch_sampler)
        assert all(len(set(ds.groups[b])) == 1 for b in batches)
        assert sorted(i for b in batches for i in b) == sorted(fold[part].tolist())
    for calib, part, pool in ((cb.val_calib, "val", "calib_val"),
                              (cb.test_calib, "test", "calib")):
        assert list(calib) == sorted(set(ds.groups[fold[part]].tolist()))
        for d, trials in calib.items():  # the pool of the session, recorded first
            rest = fold[part][ds.groups[fold[part]] == d]
            assert set(ds.groups[trials]) == {d} and max(trials) < rest.min()
        assert sorted(i for t in calib.values() for i in t) == sorted(fold[pool].tolist())
    held = set(ds.subjects[np.concatenate([fold["val"], fold["test"]])])
    assert not set(ds.subjects[fold["train"]]) & held
    assert cb.calib_sizes == [0, 4, None]


def test_other_splits_keep_the_usual_loaders():
    train, *_, cb = create_dataloaders(dict(CONFIG, split_strategy="plain"))
    assert cb is None and train.batch_size == 5


def _model():
    return SPDnet(input_dim=N_CH, hidden_layers_size=[3], output_dim=2, batchnorm=True,
                  batchnorm_momentum=0.1)


def test_statistics_are_the_batch_mean_of_the_calibration_trials():
    *_, cb = create_dataloaders(CONFIG)
    model = _model()
    d, idx = next(iter(cb.test_calib.items()))
    stats = cb._statistics(model, cb.test_calib, 4)
    layer = _bn_layers(model)[0]
    x = torch.stack([cb.dataset[i][0] for i in idx[:4]])
    with torch.no_grad():
        z = model.spdnet_layers[: list(model.spdnet_layers).index(layer)](x)
        expected = layer.mean_fun(z)
    torch.testing.assert_close(stats[d][0]["running_mean"], expected)
    assert layer.momentum == 0.1  # the model itself is untouched


def test_evaluation_restores_the_training_statistics():
    import pytorch_lightning as pl

    from spdnet_training.lightning_module import SPDNetModule

    train, val, test, n_classes, cb = create_dataloaders(CONFIG)
    module = SPDNetModule(input_dim=N_CH, hidden_layers_size=[3], output_dim=n_classes,
                          batchnorm=True, optimizer={"name": "adam", "lr": 1e-2})
    trainer = pl.Trainer(max_epochs=2, callbacks=[cb], logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False)
    trainer.fit(module, train_dataloaders=train, val_dataloaders=val)
    before = _bn_layers(module)[0].running_mean.clone()
    out = cb.evaluate(module, test)
    n_subjects = len(set(cb.subject_of.values()))
    assert set(out) == {"0", "4", "all"}
    assert all(0 <= r["accuracy"] <= 1 and len(r["subjects"]) == n_subjects
               for r in out.values())
    torch.testing.assert_close(_bn_layers(module)[0].running_mean, before)
    means = [s[0]["running_mean"] for s in cb._stats.values()]  # last size: whole pool
    assert len(means) == 2 * n_subjects and not torch.allclose(means[0], means[1])


# Every batch norm of the study: add the multivariate batch norm here once SPDnet
# accepts it. The adaptation needs a layer that normalizes with its batch statistics
# in training and whose running statistics equal them when the momentum is 1.
BATCHNORM_TYPES = ["mean_only", "mean_var_scalar"]


def _trained_model(batchnorm_type, mean_type, norm_strategy="classical"):
    """SPDnet whose batch norm has seen other data, so its state is not the initial one."""
    model = SPDnet(input_dim=N_CH, hidden_layers_size=[3], output_dim=2, batchnorm=True,
                   batchnorm_type=batchnorm_type, batchnorm_mean_type=mean_type,
                   batchnorm_norm_strategy=norm_strategy, batchnorm_momentum=0.1)
    ds = FakeSubjects()
    model.train()
    with torch.no_grad():
        for start in range(0, 40, 8):
            model(5.0 * ds.covs[start:start + 8])
    return model.eval()


@pytest.mark.parametrize("batchnorm_type", BATCHNORM_TYPES)
@pytest.mark.parametrize("mean_type", ["log_euclidean", "harmonic"])
def test_adaptation_gives_the_statistics_of_the_calibration_trials(batchnorm_type,
                                                                   mean_type):
    *_, cb = create_dataloaders(CONFIG)
    model = _trained_model(batchnorm_type, mean_type)
    stats = cb._statistics(model, cb.test_calib, 4)
    layer = _bn_layers(model)[0]
    for d, idx in cb.test_calib.items():  # every session, in turn
        x = torch.stack([cb.dataset[i][0] for i in idx[:4]])
        with torch.no_grad():
            z = model.spdnet_layers[: list(model.spdnet_layers).index(layer)](x)
            _, batch = copy.deepcopy(layer)._get_norm_parameters(z)
        batch = batch if isinstance(batch, tuple) else (batch,)
        running = [v for n, v in sorted(stats[d][0].items()) if n.startswith("running_")]
        assert len(running) == len(batch)
        for r in running:
            assert any(torch.allclose(r, b) for b in batch)


def test_minibatch_normalization_is_refused():
    *_, cb = create_dataloaders(CONFIG)
    model = _trained_model("mean_var_scalar", "log_euclidean", norm_strategy="minibatch")
    with pytest.raises(ValueError, match="classical"):
        cb._statistics(model, cb.test_calib, 4)


def test_no_batch_norm_needs_no_statistics():
    *_, cb = create_dataloaders(CONFIG)
    model = SPDnet(input_dim=N_CH, hidden_layers_size=[3], output_dim=2, batchnorm=False)
    assert cb._statistics(model, cb.test_calib, 4) is None
