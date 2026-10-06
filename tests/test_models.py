"""Model definitions: output shapes and summary features (synthetic data only)."""

import numpy as np
import pytest
import torch

from src.models import classical as C
from src.models import cnn1d, conv_autoencoder, lstm_autoencoder


def _x(b=4, t=100, f=10, seed=0):
    return torch.rand(b, t, f, generator=torch.Generator().manual_seed(seed))


def test_cnn1d_outputs_one_logit_per_window():
    net = cnn1d.build({"channels": [8, 16, 16], "kernels": [5, 5, 3], "hidden": 8, "dropout": 0.0})
    assert net(_x()).shape == (4,)


@pytest.mark.parametrize("mod, cfg", [
    (lstm_autoencoder, {"hidden": 16, "latent": 8}),
    (conv_autoencoder, {"channels": [8, 16], "latent_channels": 4}),
])
def test_autoencoders_reconstruct_input_shape_in_unit_range(mod, cfg):
    net = mod.build(cfg)
    x = _x()
    with torch.no_grad():
        y = net(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    assert lstm_autoencoder.reconstruction_score(net, x).shape == (4,)


def test_summary_features_by_hand():
    w = np.zeros((1, 4, 10), dtype=np.float32)
    w[0, :, 0] = [0.5, 0.5, 1.0, 0.0]          # IDs: A, A, B, unseen
    w[0, 1, 1] = 1.0                            # payload of the 2nd A frame differs from the 1st
    w[0, :, 9] = [0.2, 0.4, 0.6, 0.8]
    f = dict(zip(C.SUMMARY_NAMES, C.summary_features(w)[0]))
    assert f["id_entropy"] == pytest.approx(1.5)            # p = 1/2, 1/4, 1/4
    assert f["n_unique_ids"] == 3
    assert f["unseen_id_share"] == pytest.approx(0.25)
    assert f["iat_mean"] == pytest.approx(0.5)
    assert f["iat_min"] == pytest.approx(0.2)
    assert f["byte0_mean"] == pytest.approx(0.25)
    assert f["payload_change_rate"] == pytest.approx(1.0)   # one same-ID pair, changed
    assert len(f) == len(C.SUMMARY_NAMES)


def test_pos_ratio():
    assert C.pos_ratio(np.array([0, 0, 0, 1])) == 3.0
    with pytest.raises(ValueError):
        C.pos_ratio(np.zeros(3))
