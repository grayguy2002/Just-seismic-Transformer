"""PyTorch dataset for the MLAAPDE P-wave cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


PHASE_PATCH_NAMES = ["P", "Pn", "Pg", "Sg", "Sn"]
STATUS_MISSING = "PHASE_MISSING"
AUTHOR_UNKNOWN = "AUTHOR_UNKNOWN"

V21_OPTIONAL_NUMERIC_FIELDS = [
    "source_origin_uncertainty_sec",
    "source_latitude_uncertainty_km",
    "source_longitude_uncertainty_km",
    "source_depth_uncertainty_km",
    "source_magnitude_uncertainty",
    "station_local_depth_m",
    "channel_E_azimuth_deg",
    "channel_N_azimuth_deg",
    "channel_Z_azimuth_deg",
    "trace_P_arrival_sample",
    "trace_Pn_arrival_sample",
    "trace_Pg_arrival_sample",
    "trace_Sg_arrival_sample",
    "trace_Sn_arrival_sample",
]

V21_OPTIONAL_TEXT_FIELDS = [
    "source_magnitude_author",
    "selected_phase_status",
    "trace_P_status",
    "trace_Pn_status",
    "trace_Pg_status",
    "trace_Sg_status",
    "trace_Sn_status",
]

V21_FORBIDDEN_DEFAULT_CONDITION_KEYS = [
    "normalization_scale",
    "trace_snr_db",
    "event_id",
    "source_id",
    "trace_name",
    "hdf5_bucket",
    "hdf5_index",
    "cache_index",
    "waves_id",
    "phase_id",
    "source_origin_time",
    "trace_start_time",
    "selected_phase_analyst_id",
    "split",
    "subset_split",
    "month",
    "month_compact",
]


class SeismicWaveformDataset(Dataset):
    """
    Memory-mapped waveform cache + conditions table.

    Each item returns:
        waveform:  (3, n_samples) float32, stream max-abs normalized to [0, 1] → then rescaled to [-1, 1]
        conditions: dict of tensors
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "training",
        augment: bool = False,
        augment_noise_std: float = 0.01,
        augment_time_shift: int = 10,
        vocab_from: "SeismicWaveformDataset | None" = None,
        *,
        cache_prefix: str = "pwave_v1",
        condition_version: str = "legacy",
        field_policy: str = "default",
    ):
        data_dir = Path(data_dir)
        cache_dir = data_dir / "cache"
        split_dir = cache_dir / "splits"
        self.cache_prefix = cache_prefix
        self.condition_version = condition_version
        self.field_policy = field_policy
        self.is_v21 = condition_version in ("v2.1", "v3")

        self.waveforms = np.load(
            str(cache_dir / f"{cache_prefix}_X_model_20p60_streamnorm_float32.npy"),
            mmap_mode="r",
        )
        if self.waveforms.ndim != 3:
            raise ValueError(f"Expected 3-D waveform cache, got shape {self.waveforms.shape}")
        self.n_samples = self.waveforms.shape[2]

        self.conditions = pd.read_csv(cache_dir / f"{cache_prefix}_conditions.csv")
        self._prepare_conditions()

        split_path = split_dir / f"{split}_indices.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        self.indices = np.load(str(split_path)).astype(np.int64)

        if vocab_from is not None:
            self._inherit_vocabs(vocab_from)
        else:
            self._build_vocabs_and_travel_residuals()

        self._build_lookup_maps()
        self._assert_v21_condition_policy()

        self.augment = augment
        self.augment_noise_std = augment_noise_std
        self.augment_time_shift = augment_time_shift

    def _prepare_conditions(self) -> None:
        if "path_ep_distance_km" not in self.conditions.columns:
            self.conditions["path_ep_distance_km"] = self.conditions["path_ep_distance_deg"] * 111.195
        if "station_code" not in self.conditions.columns:
            self.conditions["station_code"] = "UNKNOWN"
        self.conditions["station_id"] = (
            self.conditions["station_network_code"].fillna("UNKNOWN").astype(str)
            + "."
            + self.conditions["station_code"].fillna("UNKNOWN").astype(str)
        )
        if "station_location_code" not in self.conditions.columns:
            self.conditions["station_location_code"] = "LOCATION_OTHER"
        self.conditions["station_location_code"] = (
            self.conditions["station_location_code"].fillna("LOCATION_OTHER").astype(str)
        )
        for col in V21_OPTIONAL_NUMERIC_FIELDS:
            if col not in self.conditions.columns:
                self.conditions[col] = np.nan
        for col in V21_OPTIONAL_TEXT_FIELDS:
            if col not in self.conditions.columns:
                self.conditions[col] = STATUS_MISSING if col.endswith("_status") else AUTHOR_UNKNOWN
            fill = STATUS_MISSING if col.endswith("_status") else AUTHOR_UNKNOWN
            self.conditions[col] = self.conditions[col].fillna(fill).replace("", fill).astype(str)
        if "selected_phase_arrival_sample" not in self.conditions.columns:
            self.conditions["selected_phase_arrival_sample"] = 2400.0

    def _build_tt_features(self, conditions: pd.DataFrame, enc: OneHotEncoder) -> np.ndarray:
        phase_oh = enc.transform(conditions[["selected_phase"]])
        return np.column_stack([
            conditions["path_ep_distance_deg"].values,
            conditions["path_ep_distance_deg"].values ** 2,
            np.log1p(conditions["source_depth_km"].clip(0.0).values),
            phase_oh,
        ])

    def _inherit_vocabs(self, vocab_from: "SeismicWaveformDataset") -> None:
        self._tt_reg = vocab_from._tt_reg
        self._phase_oh_encoder = vocab_from._phase_oh_encoder
        self.conditions["residual_travel_sec"] = (
            self.conditions["phase_travel_sec"].values
            - self._tt_reg.predict(self._build_tt_features(self.conditions, self._phase_oh_encoder))
        )
        self.magtype_vocab = vocab_from.magtype_vocab
        self.phase_vocab = vocab_from.phase_vocab
        self.channel_vocab = vocab_from.channel_vocab
        self.network_vocab = vocab_from.network_vocab
        self.station_id_vocab = vocab_from.station_id_vocab
        self.station_location_vocab = vocab_from.station_location_vocab
        self.source_magnitude_author_vocab = vocab_from.source_magnitude_author_vocab
        self.phase_status_vocab = vocab_from.phase_status_vocab
        self.scalar_ranges = vocab_from.scalar_ranges
        self._channel_map = vocab_from._channel_map
        self._top_nets = vocab_from._top_nets
        self._net_default_idx = vocab_from._net_default_idx
        self._station_default_idx = vocab_from._station_default_idx
        self._location_default_idx = vocab_from._location_default_idx
        self._author_default_idx = vocab_from._author_default_idx
        self._phase_status_default_idx = vocab_from._phase_status_default_idx

    def _build_vocabs_and_travel_residuals(self) -> None:
        sub = self.conditions.iloc[self.indices]
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        phase_oh = enc.fit_transform(sub[["selected_phase"]])
        X_tt = np.column_stack([
            sub["path_ep_distance_deg"].values,
            sub["path_ep_distance_deg"].values ** 2,
            np.log1p(sub["source_depth_km"].clip(0.0).values),
            phase_oh,
        ])
        y_tt = sub["phase_travel_sec"].values
        self._tt_reg = LinearRegression().fit(X_tt, y_tt)
        self._phase_oh_encoder = enc
        self.conditions["residual_travel_sec"] = (
            self.conditions["phase_travel_sec"].values
            - self._tt_reg.predict(self._build_tt_features(self.conditions, enc))
        )

        self.magtype_vocab = sorted(sub["source_magnitude_type"].dropna().unique().tolist())
        self.phase_vocab = sorted(sub["selected_phase"].dropna().unique().tolist())

        raw_channels = sub["trace_channel"].dropna().unique().tolist()
        self.channel_vocab = sorted([c for c in raw_channels if c in ("BH", "HH")])
        if any(c in raw_channels for c in ("EH", "HN", "SH")):
            self.channel_vocab.append("OTHER")
        self._channel_map = {c: c if c in ("BH", "HH") else "OTHER" for c in raw_channels}

        net_counts = sub["station_network_code"].value_counts()
        self._top_nets = set(net_counts[net_counts >= 30].index.tolist())
        self.network_vocab = sorted(self._top_nets) + ["REGIONAL_OTHER"]
        self._net_default_idx = self.network_vocab.index("REGIONAL_OTHER")

        self.station_id_vocab = sorted(sub["station_id"].dropna().unique().tolist()) + ["STATION_OTHER"]
        self._station_default_idx = self.station_id_vocab.index("STATION_OTHER")

        raw_locations = sub["station_location_code"].fillna("LOCATION_OTHER").astype(str).unique().tolist()
        self.station_location_vocab = sorted([x for x in raw_locations if x != "LOCATION_OTHER"])
        self.station_location_vocab.append("LOCATION_OTHER")
        self._location_default_idx = self.station_location_vocab.index("LOCATION_OTHER")

        authors = sub["source_magnitude_author"].fillna(AUTHOR_UNKNOWN).astype(str).unique().tolist()
        self.source_magnitude_author_vocab = sorted([x for x in authors if x != AUTHOR_UNKNOWN])
        self.source_magnitude_author_vocab.append(AUTHOR_UNKNOWN)
        self._author_default_idx = self.source_magnitude_author_vocab.index(AUTHOR_UNKNOWN)

        status_values = []
        for col in ["selected_phase_status"] + [f"trace_{p}_status" for p in PHASE_PATCH_NAMES]:
            status_values.extend(sub[col].fillna(STATUS_MISSING).astype(str).tolist())
        self.phase_status_vocab = sorted({x for x in status_values if x != STATUS_MISSING})
        self.phase_status_vocab.append(STATUS_MISSING)
        self._phase_status_default_idx = self.phase_status_vocab.index(STATUS_MISSING)

        self.scalar_ranges = self._compute_scalar_ranges(sub)

    def _compute_scalar_ranges(self, sub: pd.DataFrame) -> dict[str, list[float]]:
        fields = [
            "source_magnitude",
            "source_depth_km",
            "path_ep_distance_deg",
            "path_ep_distance_km",
            "phase_travel_sec",
            "residual_travel_sec",
            "selected_phase_arrival_sample",
            "station_elevation_m",
            *V21_OPTIONAL_NUMERIC_FIELDS,
        ]
        ranges: dict[str, list[float]] = {}
        for col in fields:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) == 0:
                ranges[col] = [0.0, 1.0]
                continue
            lo = float(vals.quantile(0.01))
            hi = float(vals.quantile(0.99))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo = float(vals.min())
                hi = float(vals.max())
            if lo == hi:
                hi = lo + 1.0
            ranges[col] = [lo, hi]
        return ranges

    def _build_lookup_maps(self) -> None:
        self.magtype_to_idx = {v: i for i, v in enumerate(self.magtype_vocab)}
        self.phase_to_idx = {v: i for i, v in enumerate(self.phase_vocab)}
        self.channel_to_idx = {v: i for i, v in enumerate(self.channel_vocab)}
        self.network_to_idx = {v: i for i, v in enumerate(self.network_vocab)}
        self.station_id_to_idx = {v: i for i, v in enumerate(self.station_id_vocab)}
        self.station_location_to_idx = {v: i for i, v in enumerate(self.station_location_vocab)}
        self.source_magnitude_author_to_idx = {v: i for i, v in enumerate(self.source_magnitude_author_vocab)}
        self.phase_status_to_idx = {v: i for i, v in enumerate(self.phase_status_vocab)}
        self._net_default_idx = self.network_to_idx["REGIONAL_OTHER"]
        self._station_default_idx = self.station_id_to_idx["STATION_OTHER"]
        self._location_default_idx = self.station_location_to_idx["LOCATION_OTHER"]
        self._author_default_idx = self.source_magnitude_author_to_idx[AUTHOR_UNKNOWN]
        self._phase_status_default_idx = self.phase_status_to_idx[STATUS_MISSING]

    def _assert_v21_condition_policy(self) -> None:
        if not self.is_v21:
            return
        emitted = set(self._base_condition_keys(include_normalization=False)) | set(self._v21_condition_keys())
        bad = emitted.intersection(V21_FORBIDDEN_DEFAULT_CONDITION_KEYS)
        if bad:
            raise ValueError(f"v2.1 default condition keys include forbidden fields: {sorted(bad)}")

    @staticmethod
    def _base_condition_keys(include_normalization: bool) -> list[str]:
        keys = [
            "source_magnitude",
            "source_depth_km",
            "path_ep_distance_deg",
            "path_ep_distance_km",
            "path_azimuth_deg",
            "path_back_azimuth_deg",
            "phase_travel_sec",
            "residual_travel_sec",
            "source_latitude_deg",
            "source_longitude_deg",
            "station_latitude_deg",
            "station_longitude_deg",
            "station_elevation_m",
            "source_magnitude_type",
            "selected_phase",
            "trace_channel",
            "station_network_code",
            "station_id",
            "station_location_code",
        ]
        if include_normalization:
            keys.append("normalization_scale")
        return keys

    @staticmethod
    def _v21_condition_keys() -> list[str]:
        keys = [
            "source_magnitude_author",
            "selected_phase_status",
            "selected_phase_arrival_sample",
            "selected_phase_arrival_sample_present",
            "augmentation_time_shift_samples",
        ]
        for col in V21_OPTIONAL_NUMERIC_FIELDS:
            keys.extend([col, f"{col}_present"])
        for phase in PHASE_PATCH_NAMES:
            keys.append(f"trace_{phase}_status")
        return keys

    @property
    def condition_spec(self) -> dict[str, Any]:
        return dict(
            magnitude_types=self.magtype_vocab,
            phases=self.phase_vocab,
            channels=self.channel_vocab,
            network_codes=self.network_vocab,
            station_ids=self.station_id_vocab,
            station_locations=self.station_location_vocab,
            source_magnitude_authors=self.source_magnitude_author_vocab,
            phase_statuses=self.phase_status_vocab,
            scalar_ranges=self.scalar_ranges,
            condition_schema_version="v2.1" if self.is_v21 else "legacy",
            field_policy_version=self.field_policy,
        )

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> tuple[float, float]:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return default, 0.0
        if not np.isfinite(val):
            return default, 0.0
        return val, 1.0

    def _numeric_tensor_with_mask(self, row: pd.Series, key: str, default: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        val, present = self._safe_float(row.get(key, np.nan), default=default)
        return torch.tensor(val, dtype=torch.float32), torch.tensor(present, dtype=torch.float32)

    def _status_idx(self, value: Any) -> int:
        if pd.isna(value):
            return self._phase_status_default_idx
        return self.phase_status_to_idx.get(str(value), self._phase_status_default_idx)

    def _author_idx(self, value: Any) -> int:
        if pd.isna(value):
            return self._author_default_idx
        return self.source_magnitude_author_to_idx.get(str(value), self._author_default_idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cache_idx = self.indices[idx]
        w = self.waveforms[cache_idx].copy().astype(np.float32)
        w = w * 2.0 - 1.0

        shift = 0
        if self.augment:
            w += np.random.randn(*w.shape).astype(np.float32) * self.augment_noise_std
            shift = int(np.random.randint(-self.augment_time_shift, self.augment_time_shift + 1))
            if shift != 0:
                w = np.roll(w, shift, axis=-1)

        row = self.conditions.iloc[cache_idx]
        cond = {
            "source_magnitude": torch.tensor(row["source_magnitude"], dtype=torch.float32),
            "source_depth_km": torch.tensor(row["source_depth_km"], dtype=torch.float32),
            "path_ep_distance_deg": torch.tensor(row["path_ep_distance_deg"], dtype=torch.float32),
            "path_ep_distance_km": torch.tensor(row["path_ep_distance_km"], dtype=torch.float32),
            "path_azimuth_deg": torch.tensor(row["path_azimuth_deg"], dtype=torch.float32),
            "path_back_azimuth_deg": torch.tensor(row["path_back_azimuth_deg"], dtype=torch.float32),
            "phase_travel_sec": torch.tensor(row["phase_travel_sec"], dtype=torch.float32),
            "residual_travel_sec": torch.tensor(row["residual_travel_sec"], dtype=torch.float32),
            "source_latitude_deg": torch.tensor(row["source_latitude_deg"], dtype=torch.float32),
            "source_longitude_deg": torch.tensor(row["source_longitude_deg"], dtype=torch.float32),
            "station_latitude_deg": torch.tensor(row["station_latitude_deg"], dtype=torch.float32),
            "station_longitude_deg": torch.tensor(row["station_longitude_deg"], dtype=torch.float32),
            "station_elevation_m": torch.tensor(row["station_elevation_m"], dtype=torch.float32),
            "source_magnitude_type": torch.tensor(
                self.magtype_to_idx.get(str(row["source_magnitude_type"]), 0), dtype=torch.long,
            ),
            "selected_phase": torch.tensor(
                self.phase_to_idx.get(str(row["selected_phase"]), 0), dtype=torch.long,
            ),
            "trace_channel": torch.tensor(
                self.channel_to_idx[self._channel_map.get(str(row["trace_channel"]), "OTHER")],
                dtype=torch.long,
            ),
            "station_network_code": torch.tensor(
                self.network_to_idx.get(str(row["station_network_code"]), self._net_default_idx),
                dtype=torch.long,
            ),
            "station_id": torch.tensor(
                self.station_id_to_idx.get(str(row["station_id"]), self._station_default_idx),
                dtype=torch.long,
            ),
            "station_location_code": torch.tensor(
                self.station_location_to_idx.get(str(row["station_location_code"]), self._location_default_idx),
                dtype=torch.long,
            ),
        }

        if not self.is_v21:
            cond["normalization_scale"] = torch.tensor(row["normalization_scale"], dtype=torch.float32)
        else:
            cond["source_magnitude_author"] = torch.tensor(
                self._author_idx(row.get("source_magnitude_author", AUTHOR_UNKNOWN)), dtype=torch.long,
            )
            cond["selected_phase_status"] = torch.tensor(
                self._status_idx(row.get("selected_phase_status", STATUS_MISSING)), dtype=torch.long,
            )
            selected_arrival, selected_arrival_present = self._numeric_tensor_with_mask(
                row, "selected_phase_arrival_sample", default=800.0,
            )
            cond["selected_phase_arrival_sample"] = selected_arrival
            cond["selected_phase_arrival_sample_present"] = selected_arrival_present
            cond["augmentation_time_shift_samples"] = torch.tensor(shift, dtype=torch.float32)
            for key in V21_OPTIONAL_NUMERIC_FIELDS:
                val, mask = self._numeric_tensor_with_mask(row, key)
                cond[key] = val
                cond[f"{key}_present"] = mask
            for phase in PHASE_PATCH_NAMES:
                cond[f"trace_{phase}_status"] = torch.tensor(
                    self._status_idx(row.get(f"trace_{phase}_status", STATUS_MISSING)),
                    dtype=torch.long,
                )

        return torch.from_numpy(w), cond


def collate_conditions(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    waveforms = torch.stack([item[0] for item in batch])
    cond_dicts = [item[1] for item in batch]
    batched_cond = {}
    for key in cond_dicts[0]:
        batched_cond[key] = torch.stack([d[key] for d in cond_dicts])
    return waveforms, batched_cond
