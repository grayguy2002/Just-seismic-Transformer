"""Seismic condition encoder with versioned condition-token layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class ScalarEncoder(nn.Module):
    """Map a single scalar to an embedding via sinusoidal frequencies."""

    def __init__(
        self,
        out_dim: int,
        freq_dim: int = 128,
        v_min: float = 0.0,
        v_max: float = 1.0,
        log_scale: bool = False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.v_min = v_min
        self.v_max = v_max
        self.log_scale = log_scale

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _normalise(self, v: torch.Tensor) -> torch.Tensor:
        if self.log_scale:
            v = torch.log1p(v.clamp(min=0.0))
        v = (v - self.v_min) / max(self.v_max - self.v_min, 1e-8)
        return v.clamp(0.0, 1.0)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        vn = self._normalise(v)
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, dtype=torch.float32, device=v.device)
            / half
        )
        args = 2.0 * math.pi * vn[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class AngleEncoder(nn.Module):
    """Map an angle in degrees to an embedding via sin/cos."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, deg: torch.Tensor) -> torch.Tensor:
        rad = torch.deg2rad(deg)
        return self.mlp(torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1))


def lat_lon_depth_to_ecef(
    lat: torch.Tensor,
    lon: torch.Tensor,
    depth_km: torch.Tensor,
) -> torch.Tensor:
    a = 6371.0
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    r = a - depth_km.clamp(min=0.0)
    x = r * torch.cos(lat_rad) * torch.cos(lon_rad)
    y = r * torch.cos(lat_rad) * torch.sin(lon_rad)
    z = r * torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)


@dataclass
class ConditionSpec:
    """Metadata needed to build a SeismicConditionEncoder."""

    magnitude_types: list[str]
    phases: list[str]
    channels: list[str]
    network_codes: list[str]
    hidden_dim: int = 768
    encoder_version: str = "v3"
    station_ids: list[str] | None = None
    station_locations: list[str] | None = None
    source_magnitude_authors: list[str] | None = None
    phase_statuses: list[str] | None = None
    scalar_ranges: dict[str, list[float]] | None = None
    condition_schema_version: str = "v2.1"
    field_policy_version: str = "default"
    use_condition_transformer: bool = False
    condition_transformer_layers: int = 0
    condition_transformer_heads: int = 4
    condition_transformer_dropout: float = 0.0

    def to_config(self) -> dict[str, Any]:
        return {
            "magnitude_types": self.magnitude_types,
            "phases": self.phases,
            "channels": self.channels,
            "network_codes": self.network_codes,
            "hidden_dim": self.hidden_dim,
            "encoder_version": self.encoder_version,
            "station_ids": self.station_ids,
            "station_locations": self.station_locations,
            "source_magnitude_authors": self.source_magnitude_authors,
            "phase_statuses": self.phase_statuses,
            "scalar_ranges": self.scalar_ranges,
            "condition_schema_version": self.condition_schema_version,
            "field_policy_version": self.field_policy_version,
            "use_condition_transformer": self.use_condition_transformer,
            "condition_transformer_layers": self.condition_transformer_layers,
            "condition_transformer_heads": self.condition_transformer_heads,
            "condition_transformer_dropout": self.condition_transformer_dropout,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ConditionSpec":
        return cls(
            magnitude_types=list(cfg["magnitude_types"]),
            phases=list(cfg["phases"]),
            channels=list(cfg["channels"]),
            network_codes=list(cfg["network_codes"]),
            hidden_dim=int(cfg.get("hidden_dim", 768)),
            encoder_version=str(cfg.get("encoder_version", "v3")),
            station_ids=cfg.get("station_ids"),
            station_locations=cfg.get("station_locations"),
            source_magnitude_authors=cfg.get("source_magnitude_authors"),
            phase_statuses=cfg.get("phase_statuses"),
            scalar_ranges=cfg.get("scalar_ranges"),
            condition_schema_version=str(cfg.get("condition_schema_version", "v2.1")),
            field_policy_version=str(cfg.get("field_policy_version", "default")),
            use_condition_transformer=bool(cfg.get("use_condition_transformer", False)),
            condition_transformer_layers=int(cfg.get("condition_transformer_layers", 0)),
            condition_transformer_heads=int(cfg.get("condition_transformer_heads", 4)),
            condition_transformer_dropout=float(cfg.get("condition_transformer_dropout", 0.0)),
        )


class SeismicConditionEncoder(nn.Module):
    """Encode seismic metadata into condition tokens."""

    def __init__(self, spec: ConditionSpec):
        super().__init__()
        self.spec = spec
        self.encoder_version = spec.encoder_version
        self.hidden_dim = spec.hidden_dim

        if self.encoder_version == "v1":
            self._init_v1(spec)
        elif self.encoder_version == "v2":
            self._init_v2(spec)
        elif self.encoder_version == "v3":
            self._init_v3(spec)
        else:
            raise ValueError(f"Unknown condition encoder version: {self.encoder_version}")

    # ------------------------------------------------------------------
    # v1: 3 tokens — source / path / receiver (one per group)
    # ------------------------------------------------------------------

    def _init_v1(self, spec: ConditionSpec) -> None:
        D = spec.hidden_dim
        self.token_names = ["source", "path", "receiver"]
        self.group_token_indices = {"source": [0], "path": [1], "receiver": [2]}
        self.n_tokens = 3

        self.mag_enc     = ScalarEncoder(128, v_min=1.0, v_max=9.0)
        self.depth_enc   = ScalarEncoder(128, v_min=0.0, v_max=6.6, log_scale=True)
        self.az_enc      = AngleEncoder(128)
        self.magtype_emb = nn.Embedding(len(spec.magnitude_types), 64)
        self.src_xyz_enc = nn.Linear(3, 128)
        self.norm_enc    = ScalarEncoder(128, v_min=3.0, v_max=17.0, log_scale=True)
        self.source_fuse = nn.Sequential(
            nn.LayerNorm(128 * 5 + 64),
            nn.Linear(128 * 5 + 64, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.dist_enc    = ScalarEncoder(128, v_min=0.0, v_max=4.65, log_scale=True)
        self.baz_enc     = AngleEncoder(128)
        self.restt_enc   = ScalarEncoder(64, v_min=-40.0, v_max=40.0)
        self.path_fuse   = nn.Sequential(
            nn.LayerNorm(128 * 2 + 64),
            nn.Linear(128 * 2 + 64, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        self.phase_emb   = nn.Embedding(len(spec.phases), 64)
        self.channel_emb = nn.Embedding(len(spec.channels), 32)
        self.network_emb = nn.Embedding(len(spec.network_codes), 64)
        self.elev_enc    = ScalarEncoder(64, v_min=-100.0, v_max=5000.0)
        self.sta_xyz_enc = nn.Linear(3, 128)
        self.receiver_fuse = nn.Sequential(
            nn.LayerNorm(64 + 32 + 64 + 64 + 128),
            nn.Linear(64 + 32 + 64 + 64 + 128, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

    def _forward_v1(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        src_xyz = self._source_xyz(
            cond["source_latitude_deg"],
            cond["source_longitude_deg"],
            cond["source_depth_km"],
        )
        s = torch.cat([
            self.mag_enc(cond["source_magnitude"]),
            self.depth_enc(cond["source_depth_km"]),
            self.az_enc(cond["path_azimuth_deg"]),
            self.magtype_emb(cond["source_magnitude_type"]),
            self.src_xyz_enc(src_xyz),
            self.norm_enc(cond["normalization_scale"]),
        ], dim=-1)
        src_tok = self.source_fuse(s)

        p = torch.cat([
            self.dist_enc(cond["path_ep_distance_deg"]),
            self.baz_enc(cond["path_back_azimuth_deg"]),
            self.restt_enc(cond["residual_travel_sec"]),
        ], dim=-1)
        path_tok = self.path_fuse(p)

        r = torch.cat([
            self.phase_emb(cond["selected_phase"]),
            self.channel_emb(cond["trace_channel"]),
            self.network_emb(cond["station_network_code"]),
            self.elev_enc(cond["station_elevation_m"]),
            self.sta_xyz_enc(self._station_xyz(cond)),
        ], dim=-1)
        rcv_tok = self.receiver_fuse(r)

        return torch.stack([src_tok, path_tok, rcv_tok], dim=1)

    # ------------------------------------------------------------------
    # v2: 10 tokens — fine-grained source / path / receiver
    # ------------------------------------------------------------------

    def _init_v2(self, spec: ConditionSpec) -> None:
        D = spec.hidden_dim
        self.token_names = [
            "source_size",
            "source_location",
            "source_radiation_proxy",
            "path_geometry",
            "path_travel_time",
            "path_phase",
            "path_region_proxy",
            "receiver_site",
            "station_identity",
            "instrument",
        ]
        self.group_token_indices = {
            "source": [0, 1, 2],
            "path": [3, 4, 5, 6],
            "receiver": [7, 8, 9],
        }
        self.n_tokens = len(self.token_names)

        station_ids = spec.station_ids or ["STATION_OTHER"]
        station_locations = spec.station_locations or ["LOCATION_OTHER"]

        self.mag_enc = ScalarEncoder(128, v_min=1.0, v_max=9.0)
        self.depth_enc = ScalarEncoder(128, v_min=0.0, v_max=6.6, log_scale=True)
        self.dist_enc = ScalarEncoder(128, v_min=0.0, v_max=4.65, log_scale=True)
        self.dist_km_enc = ScalarEncoder(128, v_min=0.0, v_max=9.4, log_scale=True)
        self.travel_time_enc = ScalarEncoder(128, v_min=0.0, v_max=7.0, log_scale=True)
        self.restt_enc = ScalarEncoder(64, v_min=-40.0, v_max=40.0)
        self.norm_enc = ScalarEncoder(128, v_min=3.0, v_max=17.0, log_scale=True)
        self.elev_enc = ScalarEncoder(64, v_min=-100.0, v_max=5000.0)
        self.az_enc = AngleEncoder(128)
        self.baz_enc = AngleEncoder(128)

        self.magtype_emb = nn.Embedding(len(spec.magnitude_types), 64)
        self.phase_emb = nn.Embedding(len(spec.phases), 64)
        self.channel_emb = nn.Embedding(len(spec.channels), 32)
        self.network_emb = nn.Embedding(len(spec.network_codes), 64)
        self.station_id_emb = nn.Embedding(len(station_ids), 128)
        self.station_location_emb = nn.Embedding(len(station_locations), 32)

        self.src_xyz_enc = nn.Linear(3, 128)
        self.sta_xyz_enc = nn.Linear(3, 128)
        self.mid_xyz_enc = nn.Linear(3, 128)
        self.delta_xyz_enc = nn.Linear(3, 128)

        self.source_size_fuse = self._fuse(128 + 64 + 128, D)
        self.source_location_fuse = self._fuse(128 + 128, D)
        self.source_radiation_proxy_fuse = self._fuse(128 + 128 + 64, D)
        self.path_geometry_fuse = self._fuse(128 + 128 + 128 + 128 + 128, D)
        self.path_travel_time_fuse = self._fuse(128 + 64 + 128 + 64, D)
        self.path_phase_fuse = self._fuse(64 + 128 + 128, D)
        self.path_region_proxy_fuse = self._fuse(128 + 128 + 128 + 128 + 64, D)
        self.receiver_site_fuse = self._fuse(128 + 64, D)
        self.station_identity_fuse = self._fuse(128 + 64, D)
        self.instrument_fuse = self._fuse(32 + 32, D)

        self._init_token_stack(spec, D)

    def _forward_v2(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        B = cond["source_magnitude"].shape[0]
        device = cond["source_magnitude"].device
        src_xyz = self._unit_xyz(self._source_xyz(
            cond["source_latitude_deg"],
            cond["source_longitude_deg"],
            cond["source_depth_km"],
        ))
        sta_xyz = self._unit_xyz(self._station_xyz(cond))
        mid_xyz = 0.5 * (src_xyz + sta_xyz)
        delta_xyz = sta_xyz - src_xyz

        dist_km = cond.get("path_ep_distance_km")
        if dist_km is None:
            dist_km = cond["path_ep_distance_deg"] * 111.195
        travel_time = cond.get("phase_travel_sec")
        if travel_time is None:
            travel_time = torch.zeros(B, device=device, dtype=cond["source_magnitude"].dtype)
        station_id = cond.get("station_id")
        if station_id is None:
            station_id = torch.zeros(B, device=device, dtype=torch.long)
        station_location = cond.get("station_location_code")
        if station_location is None:
            station_location = torch.zeros(B, device=device, dtype=torch.long)

        mag = self.mag_enc(cond["source_magnitude"])
        depth = self.depth_enc(cond["source_depth_km"])
        dist = self.dist_enc(cond["path_ep_distance_deg"])
        dist_km_emb = self.dist_km_enc(dist_km)
        az = self.az_enc(cond["path_azimuth_deg"])
        baz = self.baz_enc(cond["path_back_azimuth_deg"])
        restt = self.restt_enc(cond["residual_travel_sec"])
        tt = self.travel_time_enc(travel_time)
        norm = self.norm_enc(cond["normalization_scale"])
        elev = self.elev_enc(cond["station_elevation_m"])
        magtype = self.magtype_emb(cond["source_magnitude_type"])
        phase = self.phase_emb(cond["selected_phase"])
        channel = self.channel_emb(cond["trace_channel"])
        network = self.network_emb(cond["station_network_code"])
        station = self.station_id_emb(station_id)
        location = self.station_location_emb(station_location)
        src_pos = self.src_xyz_enc(src_xyz)
        sta_pos = self.sta_xyz_enc(sta_xyz)
        mid_pos = self.mid_xyz_enc(mid_xyz)
        delta_pos = self.delta_xyz_enc(delta_xyz)

        tokens = torch.stack([
            self.source_size_fuse(torch.cat([mag, magtype, norm], dim=-1)),
            self.source_location_fuse(torch.cat([src_pos, depth], dim=-1)),
            self.source_radiation_proxy_fuse(torch.cat([az, depth, magtype], dim=-1)),
            self.path_geometry_fuse(torch.cat([dist, dist_km_emb, az, baz, delta_pos], dim=-1)),
            self.path_travel_time_fuse(torch.cat([tt, restt, dist, phase], dim=-1)),
            self.path_phase_fuse(torch.cat([phase, dist, depth], dim=-1)),
            self.path_region_proxy_fuse(torch.cat([src_pos, sta_pos, mid_pos, delta_pos, network], dim=-1)),
            self.receiver_site_fuse(torch.cat([sta_pos, elev], dim=-1)),
            self.station_identity_fuse(torch.cat([station, network], dim=-1)),
            self.instrument_fuse(torch.cat([channel, location], dim=-1)),
        ], dim=1)
        return self._stack_tokens(tokens, device)

    # ------------------------------------------------------------------
    # v3: 11 tokens — v2.1 fields, no patches, 3 redundant tokens removed
    # ------------------------------------------------------------------

    def _init_v3(self, spec: ConditionSpec) -> None:
        D = spec.hidden_dim
        self.token_names = [
            "source_size",              # 0
            "source_location_depth",    # 1
            "source_radiation_proxy",   # 2
            "path_geometry",            # 3
            "path_travel_time",         # 4
            "selected_phase_label",     # 5
            "path_region_proxy",        # 6
            "receiver_site",            # 7
            "station_identity",         # 8
            "instrument",               # 9
            "receiver_orientation",     # 10
        ]
        self.group_token_indices = {
            "source": [0, 1, 2],
            "path": [3, 4, 5, 6],
            "receiver": [7, 8, 9, 10],
        }
        self.n_tokens = len(self.token_names)

        station_ids = spec.station_ids or ["STATION_OTHER"]
        station_locations = spec.station_locations or ["LOCATION_OTHER"]
        magnitude_authors = spec.source_magnitude_authors or ["AUTHOR_UNKNOWN"]
        phase_statuses = spec.phase_statuses or ["PHASE_MISSING"]

        # --- scalar encoders ---
        self.mag_enc = self._scalar_from_spec(spec, "source_magnitude", 128, (1.0, 9.0))
        self.depth_enc = self._scalar_from_spec(spec, "source_depth_km", 128, (0.0, 700.0), log_scale=True)
        self.dist_enc = self._scalar_from_spec(spec, "path_ep_distance_deg", 128, (0.0, 10.0))
        self.dist_km_enc = self._scalar_from_spec(spec, "path_ep_distance_km", 128, (0.0, 1200.0))
        self.travel_time_enc = self._scalar_from_spec(spec, "phase_travel_sec", 128, (0.0, 120.0))
        self.restt_enc = self._scalar_from_spec(spec, "residual_travel_sec", 64, (-40.0, 40.0))
        self.elev_enc = self._scalar_from_spec(spec, "station_elevation_m", 64, (-100.0, 5000.0))
        self.local_depth_enc = self._scalar_from_spec(spec, "station_local_depth_m", 64, (0.0, 1000.0))
        self.origin_unc_enc = self._scalar_from_spec(spec, "source_origin_uncertainty_sec", 64, (0.0, 10.0))
        self.lat_unc_enc = self._scalar_from_spec(spec, "source_latitude_uncertainty_km", 64, (0.0, 50.0))
        self.lon_unc_enc = self._scalar_from_spec(spec, "source_longitude_uncertainty_km", 64, (0.0, 50.0))
        self.depth_unc_enc = self._scalar_from_spec(spec, "source_depth_uncertainty_km", 64, (0.0, 50.0))
        self.mag_unc_enc = self._scalar_from_spec(spec, "source_magnitude_uncertainty", 64, (0.0, 2.0))
        self.selected_arrival_enc = self._scalar_from_spec(
            spec, "selected_phase_arrival_sample", 64, (0.0, 4800.0)
        )
        self.az_enc = AngleEncoder(128)
        self.baz_enc = AngleEncoder(128)
        self.orientation_angle_enc = AngleEncoder(64)

        # --- categorical embeddings ---
        self.magtype_emb = nn.Embedding(len(spec.magnitude_types), 64)
        self.phase_emb = nn.Embedding(len(spec.phases), 64)
        self.channel_emb = nn.Embedding(len(spec.channels), 32)
        self.network_emb = nn.Embedding(len(spec.network_codes), 64)
        self.station_id_emb = nn.Embedding(len(station_ids), 128)
        self.station_location_emb = nn.Embedding(len(station_locations), 32)
        self.source_magnitude_author_emb = nn.Embedding(len(magnitude_authors), 64)
        self.phase_status_emb = nn.Embedding(len(phase_statuses), 64)

        # --- spatial projections ---
        self.src_xyz_enc = nn.Linear(3, 128)
        self.sta_xyz_enc = nn.Linear(3, 128)
        self.mid_xyz_enc = nn.Linear(3, 128)
        self.delta_xyz_enc = nn.Linear(3, 128)

        # --- mask encoders ---
        self.one_mask_enc = nn.Linear(1, 32)
        self.location_unc_mask_enc = nn.Linear(4, 64)
        self.uncertainty_mask_enc = nn.Linear(5, 64)
        self.orientation_mask_enc = nn.Linear(3, 64)

        # --- per-token fusion modules ---
        self.source_size_fuse = self._fuse(128 + 64 + 64 + 64 + 32, D)
        self.source_location_depth_fuse = self._fuse(128 + 128 + 64 * 4 + 64, D)
        self.source_radiation_proxy_fuse = self._fuse(128 + 128 + 64 + 128, D)
        self.path_geometry_fuse = self._fuse(128 + 128 + 128 + 128 + 128, D)
        self.path_travel_time_fuse = self._fuse(128 + 64 + 128 + 64, D)
        self.selected_phase_label_fuse = self._fuse(64 + 64 + 64 + 32, D)
        self.path_region_proxy_fuse = self._fuse(128 + 128 + 128 + 128 + 64, D)
        self.receiver_site_fuse = self._fuse(128 + 64 + 64 + 32, D)
        self.station_identity_fuse = self._fuse(128 + 64 + 32, D)
        self.instrument_fuse = self._fuse(32 + 32, D)
        self.receiver_orientation_fuse = self._fuse(64 * 3 + 64, D)

        self._init_token_stack(spec, D)

    def _forward_v3(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        ref = cond["source_magnitude"]
        device = ref.device

        # --- geometry ---
        src_xyz = self._unit_xyz(self._source_xyz(
            cond["source_latitude_deg"],
            cond["source_longitude_deg"],
            cond["source_depth_km"],
        ))
        sta_xyz = self._unit_xyz(self._station_xyz(cond))
        mid_xyz = 0.5 * (src_xyz + sta_xyz)
        delta_xyz = sta_xyz - src_xyz

        # --- fallbacks ---
        dist_km = cond.get("path_ep_distance_km")
        if dist_km is None:
            dist_km = cond["path_ep_distance_deg"] * 111.195
        travel_time = cond.get("phase_travel_sec")
        if travel_time is None:
            travel_time = torch.zeros_like(ref)

        # --- optional fields with presence masks ---
        origin_unc, origin_unc_m = self._optional_value(cond, "source_origin_uncertainty_sec", ref)
        lat_unc, lat_unc_m = self._optional_value(cond, "source_latitude_uncertainty_km", ref)
        lon_unc, lon_unc_m = self._optional_value(cond, "source_longitude_uncertainty_km", ref)
        depth_unc, depth_unc_m = self._optional_value(cond, "source_depth_uncertainty_km", ref)
        mag_unc, mag_unc_m = self._optional_value(cond, "source_magnitude_uncertainty", ref)
        local_depth, local_depth_m = self._optional_value(cond, "station_local_depth_m", ref)
        e_az, e_az_m = self._optional_value(cond, "channel_E_azimuth_deg", ref)
        n_az, n_az_m = self._optional_value(cond, "channel_N_azimuth_deg", ref)
        z_az, z_az_m = self._optional_value(cond, "channel_Z_azimuth_deg", ref)
        selected_arrival, selected_arrival_m = self._optional_value(
            cond, "selected_phase_arrival_sample", ref, 800.0,
        )

        # --- encode scalars ---
        mag = self.mag_enc(ref)
        depth = self.depth_enc(cond["source_depth_km"])
        dist = self.dist_enc(cond["path_ep_distance_deg"])
        dist_km_emb = self.dist_km_enc(dist_km)
        az = self.az_enc(cond["path_azimuth_deg"])
        baz = self.baz_enc(cond["path_back_azimuth_deg"])
        restt = self.restt_enc(cond["residual_travel_sec"])
        tt = self.travel_time_enc(travel_time)
        elev = self.elev_enc(cond["station_elevation_m"])
        local_depth_emb = self.local_depth_enc(local_depth)
        origin_unc_emb = self.origin_unc_enc(origin_unc)
        lat_unc_emb = self.lat_unc_enc(lat_unc)
        lon_unc_emb = self.lon_unc_enc(lon_unc)
        depth_unc_emb = self.depth_unc_enc(depth_unc)
        mag_unc_emb = self.mag_unc_enc(mag_unc)
        selected_arrival_emb = self.selected_arrival_enc(selected_arrival)

        # --- encode masks ---
        mag_unc_mask = self.one_mask_enc(mag_unc_m[:, None])
        local_depth_mask = self.one_mask_enc(local_depth_m[:, None])
        selected_arrival_mask = self.one_mask_enc(selected_arrival_m[:, None])
        location_unc_mask = self.location_unc_mask_enc(torch.stack([
            origin_unc_m, lat_unc_m, lon_unc_m, depth_unc_m,
        ], dim=-1))
        uncertainty_mask = self.uncertainty_mask_enc(torch.stack([
            origin_unc_m, lat_unc_m, lon_unc_m, depth_unc_m, mag_unc_m,
        ], dim=-1))
        orientation_mask = self.orientation_mask_enc(torch.stack([e_az_m, n_az_m, z_az_m], dim=-1))

        # --- embeddings ---
        magtype = self.magtype_emb(cond["source_magnitude_type"])
        phase = self.phase_emb(cond["selected_phase"])
        channel = self.channel_emb(cond["trace_channel"])
        network = self.network_emb(cond["station_network_code"])
        station = self.station_id_emb(self._optional_index(cond, "station_id", ref))
        location = self.station_location_emb(self._optional_index(cond, "station_location_code", ref))
        author = self.source_magnitude_author_emb(
            self._optional_index(cond, "source_magnitude_author", ref)
        )
        selected_status = self.phase_status_emb(
            self._optional_index(cond, "selected_phase_status", ref)
        )

        # --- spatial embeddings ---
        src_pos = self.src_xyz_enc(src_xyz)
        sta_pos = self.sta_xyz_enc(sta_xyz)
        mid_pos = self.mid_xyz_enc(mid_xyz)
        delta_pos = self.delta_xyz_enc(delta_xyz)

        # --- orientation ---
        orientation = torch.cat([
            self.orientation_angle_enc(e_az),
            self.orientation_angle_enc(n_az),
            self.orientation_angle_enc(z_az),
            orientation_mask,
        ], dim=-1)

        # --- assemble 11 tokens ---
        tokens = torch.stack([
            # 0: source_size
            self.source_size_fuse(torch.cat([mag, magtype, mag_unc_emb, author, mag_unc_mask], dim=-1)),
            # 1: source_location_depth
            self.source_location_depth_fuse(torch.cat([
                src_pos, depth, origin_unc_emb, lat_unc_emb, lon_unc_emb,
                depth_unc_emb, location_unc_mask,
            ], dim=-1)),
            # 2: source_radiation_proxy
            self.source_radiation_proxy_fuse(torch.cat([az, depth, magtype, mag], dim=-1)),
            # 3: path_geometry
            self.path_geometry_fuse(torch.cat([dist, dist_km_emb, az, baz, delta_pos], dim=-1)),
            # 4: path_travel_time
            self.path_travel_time_fuse(torch.cat([tt, restt, dist, phase], dim=-1)),
            # 5: selected_phase_label
            self.selected_phase_label_fuse(torch.cat([
                phase, selected_status, selected_arrival_emb, selected_arrival_mask,
            ], dim=-1)),
            # 6: path_region_proxy
            self.path_region_proxy_fuse(torch.cat([src_pos, sta_pos, mid_pos, delta_pos, network], dim=-1)),
            # 7: receiver_site
            self.receiver_site_fuse(torch.cat([sta_pos, elev, local_depth_emb, local_depth_mask], dim=-1)),
            # 8: station_identity
            self.station_identity_fuse(torch.cat([station, network, location], dim=-1)),
            # 9: instrument
            self.instrument_fuse(torch.cat([channel, location], dim=-1)),
            # 10: receiver_orientation
            self.receiver_orientation_fuse(orientation),
        ], dim=1)
        return self._stack_tokens(tokens, device)

    # ------------------------------------------------------------------
    # Shared infrastructure
    # ------------------------------------------------------------------

    def _init_token_stack(self, spec: ConditionSpec, hidden_dim: int) -> None:
        self.token_type_embed = nn.Embedding(self.n_tokens, hidden_dim)
        use_transformer = spec.use_condition_transformer and spec.condition_transformer_layers > 0
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=spec.condition_transformer_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=spec.condition_transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.condition_transformer = nn.TransformerEncoder(
                layer,
                num_layers=spec.condition_transformer_layers,
            )
        else:
            self.condition_transformer = None
        self.final_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _fuse(in_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _scalar_from_spec(
        spec: ConditionSpec,
        key: str,
        out_dim: int,
        default: tuple[float, float],
        *,
        log_scale: bool = False,
    ) -> ScalarEncoder:
        lo, hi = default
        if spec.scalar_ranges and key in spec.scalar_ranges:
            lo, hi = [float(x) for x in spec.scalar_ranges[key]]
        if log_scale:
            lo = math.log1p(max(lo, 0.0))
            hi = math.log1p(max(hi, 0.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo, hi = default
            if log_scale:
                lo = math.log1p(max(lo, 0.0))
                hi = math.log1p(max(hi, 0.0))
        return ScalarEncoder(out_dim, v_min=lo, v_max=hi, log_scale=log_scale)

    def _source_xyz(self, lat, lon, depth):
        return lat_lon_depth_to_ecef(lat, lon, depth)

    def _station_xyz(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        B = cond["station_latitude_deg"].shape[0]
        return lat_lon_depth_to_ecef(
            cond["station_latitude_deg"],
            cond["station_longitude_deg"],
            torch.zeros(B, device=cond["station_latitude_deg"].device),
        )

    @staticmethod
    def _unit_xyz(x: torch.Tensor) -> torch.Tensor:
        return x / 6371.0

    @staticmethod
    def _optional_value(
        cond: dict[str, torch.Tensor],
        key: str,
        ref: torch.Tensor,
        default: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key not in cond:
            value = torch.full_like(ref, default)
            present = torch.zeros_like(ref)
            return value, present
        value = cond[key].to(device=ref.device, dtype=ref.dtype)
        present_key = f"{key}_present"
        if present_key in cond:
            present = cond[present_key].to(device=ref.device, dtype=ref.dtype)
        else:
            present = torch.ones_like(ref)
        value = torch.where(torch.isfinite(value), value, torch.full_like(value, default))
        return value, present

    @staticmethod
    def _optional_index(
        cond: dict[str, torch.Tensor],
        key: str,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if key not in cond:
            return torch.zeros(ref.shape[0], device=ref.device, dtype=torch.long)
        return cond[key].to(device=ref.device, dtype=torch.long)

    def _stack_tokens(self, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
        token_ids = torch.arange(self.n_tokens, device=device)
        tokens = tokens + self.token_type_embed(token_ids).unsqueeze(0)
        if self.condition_transformer is not None:
            tokens = tokens + self.condition_transformer(tokens)
        return self.final_norm(tokens)

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.encoder_version == "v1":
            return self._forward_v1(cond)
        if self.encoder_version == "v2":
            return self._forward_v2(cond)
        return self._forward_v3(cond)
