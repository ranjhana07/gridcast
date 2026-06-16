import numpy as np
import pytest
import torch

from gridcast.models.deep import DeepConfig, EncoderDecoder, QuantileSeq2Seq, pinball_loss

SMALL = DeepConfig(lookback=48, hidden=8, epochs=2, batch_size=128, threads=2)


def test_pinball_loss_is_zero_for_perfect_prediction():
    target = torch.tensor([[1.0, 2.0]])
    prediction = target.unsqueeze(-1).repeat(1, 1, 3)
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    assert pinball_loss(prediction, target, quantiles).item() == 0.0


def test_network_outputs_ordered_quantiles():
    network = EncoderDecoder(5, 7, 8)
    out = network(torch.randn(4, 12, 5), torch.randn(4, 6, 7))
    assert out.shape == (4, 6, 3)
    assert (out[..., 0] <= out[..., 1]).all()
    assert (out[..., 1] <= out[..., 2]).all()


def test_fit_and_predict_shapes(synthetic_frame):
    model = QuantileSeq2Seq(horizon=6, config=SMALL).fit(synthetic_frame.iloc[: 24 * 60])
    origins = synthetic_frame.index[24 * 62 : 24 * 64 : 6]
    forecast = model.predict(synthetic_frame, origins)
    assert forecast.point.shape == (len(origins), 6)
    assert set(forecast.quantiles) == {0.1, 0.5, 0.9}
    assert (forecast.quantiles[0.9] >= forecast.quantiles[0.1]).all()
    assert len(model.history) == SMALL.epochs


def test_training_is_deterministic(synthetic_frame):
    train = synthetic_frame.iloc[: 24 * 40]
    origins = synthetic_frame.index[24 * 42 : 24 * 43 : 6]
    first = QuantileSeq2Seq(horizon=4, config=SMALL).fit(train).predict(synthetic_frame, origins)
    second = QuantileSeq2Seq(horizon=4, config=SMALL).fit(train).predict(synthetic_frame, origins)
    assert np.allclose(first.point, second.point, atol=1e-4)


def test_predict_before_fit_fails(synthetic_frame):
    with pytest.raises(RuntimeError):
        QuantileSeq2Seq(horizon=4, config=SMALL).predict(synthetic_frame, synthetic_frame.index[[300]])
