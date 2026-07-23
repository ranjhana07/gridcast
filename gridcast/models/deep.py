from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from gridcast.config import worker_threads
from gridcast.models.base import Forecast, origin_positions

ENCODER_COVARIATES = (
    "temperature_2m",
    "shortwave_radiation",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
)
DECODER_COVARIATES = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
)
QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class DeepConfig:
    lookback: int = 48
    hidden: int = 32
    epochs: int = 6
    samples_per_epoch: int = 6000
    batch_size: int = 256
    learning_rate: float = 2e-3
    validation_share: float = 0.1
    seed: int = 7
    threads: int = field(default_factory=worker_threads)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    diff = target.unsqueeze(-1) - prediction
    return torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean()


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_inputs: int, decoder_inputs: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(encoder_inputs, hidden, batch_first=True)
        self.decoder = nn.LSTM(decoder_inputs, hidden, batch_first=True)
        self.median = nn.Linear(hidden, 1)
        self.spread = nn.Linear(hidden, 2)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(encoder_input)
        output, _ = self.decoder(decoder_input, state)
        median = self.median(output)
        spread = nn.functional.softplus(self.spread(output))
        lower = median - spread[..., :1]
        upper = median + spread[..., 1:]
        return torch.cat([lower, median, upper], dim=-1)


class QuantileSeq2Seq:
    def __init__(self, target: str = "load_kw", horizon: int = 24, config: DeepConfig | None = None) -> None:
        self.target = target
        self.horizon = horizon
        self.config = config or DeepConfig()
        self.name = "lstm_quantile"
        self.network: EncoderDecoder | None = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.encoder_stats: tuple[np.ndarray, np.ndarray] | None = None
        self.decoder_stats: tuple[np.ndarray, np.ndarray] | None = None
        self.history: list[dict[str, float]] = []

    def covariates(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.encoder_stats is not None and self.decoder_stats is not None
        target = ((frame[self.target].to_numpy() - self.target_mean) / self.target_std).astype(np.float32)
        encoder = frame[list(ENCODER_COVARIATES)].to_numpy(dtype=np.float32)
        decoder = frame[list(DECODER_COVARIATES)].to_numpy(dtype=np.float32)
        encoder = (encoder - self.encoder_stats[0]) / self.encoder_stats[1]
        decoder = (decoder - self.decoder_stats[0]) / self.decoder_stats[1]
        return target, encoder.astype(np.float32), decoder.astype(np.float32)

    def batch(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        encoder: np.ndarray,
        decoder: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lookback = np.arange(-self.config.lookback + 1, 1)
        ahead = np.arange(1, self.horizon + 1)
        past = positions[:, None] + lookback[None, :]
        future = positions[:, None] + ahead[None, :]
        encoder_input = np.concatenate([target[past][..., None], encoder[past]], axis=-1)
        daily = target[future - 24][..., None]
        weekly = target[future - 168][..., None]
        decoder_input = np.concatenate([decoder[future], daily, weekly], axis=-1)
        return torch.from_numpy(encoder_input), torch.from_numpy(decoder_input)

    def fit(self, frame: pd.DataFrame) -> "QuantileSeq2Seq":
        config = self.config
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.set_num_threads(config.threads)
        values = frame[self.target].to_numpy()
        self.target_mean = float(values.mean())
        self.target_std = float(values.std())
        enc = frame[list(ENCODER_COVARIATES)].to_numpy(dtype=np.float32)
        dec = frame[list(DECODER_COVARIATES)].to_numpy(dtype=np.float32)
        self.encoder_stats = (enc.mean(0), enc.std(0) + 1e-6)
        self.decoder_stats = (dec.mean(0), dec.std(0) + 1e-6)
        target, encoder, decoder = self.covariates(frame)

        first = max(config.lookback, 168)
        positions = np.arange(first, len(frame) - self.horizon)
        split = int(len(positions) * (1 - config.validation_share))
        train_positions, valid_positions = positions[:split], positions[split:]

        self.network = EncoderDecoder(1 + len(ENCODER_COVARIATES), len(DECODER_COVARIATES) + 2, config.hidden)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        quantiles = torch.tensor(QUANTILES, dtype=torch.float32)
        best_state = copy.deepcopy(self.network.state_dict())
        best_loss = float("inf")
        rng = np.random.default_rng(config.seed)
        ahead = np.arange(1, self.horizon + 1)

        for epoch in range(config.epochs):
            self.network.train()
            order = rng.permutation(train_positions)[: config.samples_per_epoch]
            running = 0.0
            batches = 0
            for start in range(0, len(order), config.batch_size):
                chunk = order[start : start + config.batch_size]
                enc_in, dec_in = self.batch(chunk, target, encoder, decoder)
                truth = torch.from_numpy(target[chunk[:, None] + ahead[None, :]])
                optimiser.zero_grad()
                loss = pinball_loss(self.network(enc_in, dec_in), truth, quantiles)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimiser.step()
                running += loss.item()
                batches += 1
            valid_loss = self.evaluate(valid_positions, target, encoder, decoder, quantiles)
            self.history.append({"epoch": epoch + 1, "train_loss": running / batches, "valid_loss": valid_loss})
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(self.network.state_dict())
        self.network.load_state_dict(best_state)
        return self

    def evaluate(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        encoder: np.ndarray,
        decoder: np.ndarray,
        quantiles: torch.Tensor,
    ) -> float:
        assert self.network is not None
        self.network.eval()
        ahead = np.arange(1, self.horizon + 1)
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(positions), 1024):
                chunk = positions[start : start + 1024]
                enc_in, dec_in = self.batch(chunk, target, encoder, decoder)
                truth = torch.from_numpy(target[chunk[:, None] + ahead[None, :]])
                total += float(pinball_loss(self.network(enc_in, dec_in), truth, quantiles)) * len(chunk)
                count += len(chunk)
        return total / count

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        if self.network is None:
            raise RuntimeError("model has not been fitted")
        torch.set_num_threads(self.config.threads)
        positions = origin_positions(frame, origins)
        target, encoder, decoder = self.covariates(frame)
        self.network.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(positions), 1024):
                chunk = positions[start : start + 1024]
                enc_in, dec_in = self.batch(chunk, target, encoder, decoder)
                outputs.append(self.network(enc_in, dec_in).numpy())
        scaled = np.concatenate(outputs, axis=0)
        restored = scaled * self.target_std + self.target_mean
        return Forecast(
            point=restored[..., 1],
            quantiles={level: restored[..., index] for index, level in enumerate(QUANTILES)},
        )
