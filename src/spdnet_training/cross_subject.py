"""Cross-subject protocol: batch norm adapted to each session of each subject.

A split named ``subject_fold_<k>`` (spdnet-datasets, ``subject_folds``) holds
training subjects (``train``), validation subjects (``calib_val``, ``val``) and test
subjects (``calib``, ``test``), never shared. Each session of a validation or test
subject has an unlabeled calibration pool, recorded before its evaluated trials.
The unit of adaptation is a session of a subject, a *domain*: the ``groups`` of the
dataset (``"<subject>-<session>"``). With such a split, :func:`create_dataloaders`
returns

- a training loader whose batches each hold trials of one domain, so the batch
  norm is trained with the statistics of one domain at a time;
- validation and test loaders with one batch per domain;
- a :class:`DomainBatchNorm` callback. Before each evaluated domain, it sets the
  running statistics of every SPD batch norm to those of this domain, computed on
  its calibration pool: the whole pool for a validation domain, its first ``K``
  trials for a test domain. ``K = 0`` keeps the running statistics of training
  (the moving average of the training batches): no adaptation.

The statistics come from a forward pass with momentum 1 on a copy of the network.
They are exact for the first batch norm. A later batch norm sees the earlier ones in
training mode. The layer must set its running statistics to the batch statistics
when the momentum is 1, and normalize with its batch statistics in training: the
``minibatch`` strategy of yetanotherspdnet, which normalizes with a mean averaged
over batches, is refused.
"""

import copy
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from spdnet_datasets import DatasetManager
from torch.utils.data import DataLoader, Sampler

PREFIX = "subject_fold_"
log = logging.getLogger(__name__)


def _bn_layers(module):
    from yetanotherspdnet.nn.batchnorm import BatchNormSPDBase

    return [m for m in module.modules() if isinstance(m, BatchNormSPDBase)]


def _buffers(layers):
    return [{n: b.clone() for n, b in m.named_buffers()} for m in layers]


def _by_domain(indices, domains):
    """``{domain: [trial indices]}``, domains sorted, trials in the given order."""
    out = {}
    for i in indices:
        out.setdefault(str(domains[i]), []).append(int(i))
    return dict(sorted(out.items()))


class DomainBatchSampler(Sampler):
    """Batches of about ``batch_size`` trials of one domain, reshuffled every epoch.

    The trials of a domain are cut into ``ceil(n / batch_size)`` batches of nearly
    equal size, so that no batch is much smaller than the others. The order depends
    on ``seed`` and on the epoch, set by Lightning through :meth:`set_epoch` (it
    calls ``batch_sampler.sampler.set_epoch``), so it does not depend on the number
    of workers.
    """

    def __init__(self, indices, domains, batch_size, seed=0):
        self.groups = list(_by_domain(indices, domains).values())
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.sampler = self  # where Lightning looks for set_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _n_batches(self, group):
        return -(-len(group) // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        batches = [b.tolist() for g in self.groups
                   for b in np.array_split(rng.permutation(g), self._n_batches(g))]
        return iter([batches[i] for i in rng.permutation(len(batches))])

    def __len__(self):
        return sum(self._n_batches(g) for g in self.groups)


class DomainBatchNorm(pl.Callback):
    """Batch-norm statistics of each evaluated domain (module docstring).

    Parameters
    ----------
    dataset : the dataset of the loaders (``dataset[i] = (covariance, label)``).
    val_calib, test_calib : dict
        ``{domain: calibration trials}``, in the order of the batches of the
        validation and test loaders.
    subject_of : dict
        ``{domain: subject}``, to report the accuracy of each test subject.
    calib_sizes : list
        Calibration sizes ``K`` of :meth:`evaluate`: ``0`` keeps the running
        statistics of training, ``None`` uses the whole pool.
    """

    def __init__(self, dataset, val_calib, test_calib, subject_of, calib_sizes=(None,)):
        self.dataset, self.val_calib, self.test_calib = dataset, val_calib, test_calib
        self.subject_of = subject_of
        self.calib_sizes, self.calib_size = list(calib_sizes), None
        self.domain_counts = {}

    def _statistics(self, model, calib, size):
        """``{domain: buffers of each batch norm}``, or None for ``size == 0``."""
        if size == 0 or not _bn_layers(model):
            return None
        if size == 1:
            raise ValueError("A calibration needs at least two trials (or K = 0)")
        if any(getattr(m, "norm_strategy", "classical") != "classical"
               for m in _bn_layers(model)):
            raise ValueError("Batch norm adapted to each domain: use "
                             "batchnorm_norm_strategy=classical")
        if size is not None and size > min(len(idx) for idx in calib.values()):
            log.warning(f"K={size} exceeds the smallest calibration pool "
                        f"({min(len(idx) for idx in calib.values())} trials)")
        net = copy.deepcopy(model).eval()
        layers = _bn_layers(net)
        for m in layers:
            m.momentum = 1.0
            m.train()
        ref = next(net.parameters())
        out = {}
        with torch.no_grad():
            for d, idx in calib.items():
                x = torch.stack([self.dataset[i][0] for i in idx[:size]])
                net(x.to(dtype=ref.dtype, device=ref.device))
                out[d] = _buffers(layers)
        return out

    @staticmethod
    def _load(pl_module, buffers):
        for m, saved in zip(_bn_layers(pl_module), buffers, strict=True):
            for n, b in m.named_buffers():
                b.copy_(saved[n])

    def _start(self, pl_module, calib, size):
        self._saved = _buffers(_bn_layers(pl_module))
        self._stats = self._statistics(pl_module.model, calib, size)
        self._order = list(calib)

    def _batch(self, pl_module, batch_idx):
        if self._stats is not None:
            self._load(pl_module, self._stats[self._order[batch_idx]])

    def on_validation_epoch_start(self, trainer, pl_module):
        self._start(pl_module, self.val_calib, None)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx,
                                  dataloader_idx=0):
        self._batch(pl_module, batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._load(pl_module, self._saved)

    def on_test_epoch_start(self, trainer, pl_module):
        self._start(pl_module, self.test_calib, self.calib_size)
        self.domain_counts = {}

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx,
                            dataloader_idx=0):
        self._batch(pl_module, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloader_idx=0):
        x, y = batch
        with torch.no_grad():
            correct = (pl_module(x).argmax(dim=1) == y).sum().item()
        self.domain_counts[self._order[batch_idx]] = (correct, len(y))

    def on_test_epoch_end(self, trainer, pl_module):
        self._load(pl_module, self._saved)

    def _subject_accuracy(self):
        counts = {}
        for d, (c, n) in self.domain_counts.items():
            s = str(self.subject_of[d])
            counts[s] = np.add(counts.get(s, (0, 0)), (c, n))
        return {s: c / n for s, (c, n) in counts.items()}

    def evaluate(self, model, test_loader, ckpt_path=None, **trainer_kwargs):
        """Test accuracy for each calibration size of :attr:`calib_sizes`.

        Runs on a separate trainer, so the callbacks of training see none of these
        tests. Returns ``{K: {"accuracy", "subject_mean", "subjects"}}``, with
        ``K = "all"`` for the whole pool and the accuracy of each test subject.
        """
        tester = pl.Trainer(callbacks=[self], logger=False, enable_checkpointing=False,
                            enable_progress_bar=False, enable_model_summary=False,
                            **trainer_kwargs)
        out = {}
        for size in self.calib_sizes:
            self.calib_size = size
            res = tester.test(model, dataloaders=test_loader, ckpt_path=ckpt_path or None,
                              verbose=False)[0]
            acc = self._subject_accuracy()
            out["all" if size is None else str(size)] = {
                "accuracy": res["test/accuracy"],
                "subject_mean": float(np.mean(list(acc.values()))),
                "subjects": acc,
            }
        self.calib_size = None
        return out


def create_dataloaders(config):
    """``DatasetManager.create_dataloaders`` with the protocol of a ``subject_fold_<k>`` split.

    The config key ``calib_sizes`` (default ``[None]``) is read here and not passed
    to the dataset. Returns ``(train, val, test, num_classes, callback)``, with
    ``callback = None`` for any other split.
    """
    config = dict(config)
    calib_sizes = config.pop("calib_sizes", None) or [None]
    train, val, test, num_classes = DatasetManager.create_dataloaders(config)
    split = config.get("split_strategy")
    if not (isinstance(split, str) and split.startswith(PREFIX)):
        return train, val, test, num_classes, None

    dataset = train.dataset.dataset  # the full dataset behind the Subset of the split
    parts, domains = dataset.splits[split], dataset.groups
    val_trials = _by_domain(parts["val"], domains)
    test_trials = _by_domain(parts["test"], domains)
    pools = {**_by_domain(parts["calib_val"], domains),
             **_by_domain(parts["calib"], domains)}
    workers = {"num_workers": config.get("num_workers", 0)}
    sampler = DomainBatchSampler(parts["train"], domains, config.get("batch_size", 32),
                                 seed=config.get("seed", 42))
    loaders = [DataLoader(dataset, batch_sampler=sampler, **workers)]
    loaders += [DataLoader(dataset, batch_sampler=list(t.values()), **workers)
                for t in (val_trials, test_trials)]
    callback = DomainBatchNorm(
        dataset,
        val_calib={d: pools[d] for d in val_trials},
        test_calib={d: pools[d] for d in test_trials},
        subject_of={d: dataset.subjects[t[0]] for d, t in test_trials.items()},
        calib_sizes=calib_sizes,
    )
    return *loaders, num_classes, callback
