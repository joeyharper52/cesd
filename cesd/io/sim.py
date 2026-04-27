"""
cesd.io.sim — CESD v0.4 Phase A simulation profile loader.

Produces (n_e(r), T(r)) CSV pairs in the X-COP schema from simulation
snapshots, plus a JSON metadata sidecar for reproducibility.

The locked v0.2 detector spec is NOT modified by this module. Outputs conform
to the existing cesd.from_csv schema and ingest through the unchanged
detector pipeline.

Two orchestrators:

```
sim_to_csv_yt_native(snapshot, out_dir)
    Reads a snapshot via yt and bins it with yt.create_profile, which
    handles AMR cell-to-bin geometry correctly via internal volume
    weighting. **This is the validated path for AMR data** (Phase A
    checkpoint, Addendum A in Volume 3 Chapter 2).

sim_to_csv(snapshot, out_dir, cell_data=...)
    Bins pre-extracted per-cell arrays via the pure-NumPy
    bin_profiles_from_cells helper. Use this for synthetic input or
    for non-yt sources where you already have cell-level data.
    NOT recommended for AMR data — see Addendum A for why.
```

Output schema (unchanged across v0.1 and v0.2):

```
<base>_ne.csv       columns: r_kpc, ne_cm3, n_cells
<base>_T.csv        columns: r_kpc, T_keV, T_sl_keV, T_ew_keV, T_mw_keV
                    (T_keV is a copy of the locked primary weighting,
                    default T_sl_keV)
<base>_meta.json    reproducibility sidecar (see write_metadata for fields)
```

Three temperature weightings are computed and stored for every snapshot:

```
T_mw   mass-weighted,                      weight = m
T_ew   emission-weighted (bremsstrahlung proxy),  weight = m * rho
T_sl   spectroscopic-like (Mazzotta+04),    weight = m * rho * T^(-3/4)
```

Phase B will run the locked detector on T_sl as primary input and on T_mw as
the comparison baseline. T_ew is stored as a sensitivity diagnostic.

Author: Joey Harper, Independent Researcher.
“””

from **future** import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOADER_VERSION = “0.2.0”

# —————————————————————————

# Physical constants (CGS)

# —————————————————————————

M_PROTON_G = 1.6726219e-24
CM_PER_KPC = 3.0857e21
KEV_PER_K = 8.617333262e-8  # k_B in keV/K

# Mean molecular weight per electron for X=0.7, Y=0.28, Z=0.02 fully ionized

MU_E_DEFAULT = 1.17

_ALLOWED_CENTERING = (“potential_min”, “density_peak”)
_ALLOWED_PRIMARY_T = (“spectroscopic_like”, “emission_weighted”, “mass_weighted”)
_PRIMARY_T_KEY_MAP = {
“spectroscopic_like”: “T_sl_keV”,
“emission_weighted”: “T_ew_keV”,
“mass_weighted”: “T_mw_keV”,
}

# —————————————————————————

# Configuration

# —————————————————————————

@dataclass
class SimSnapshot:
“”“Configuration for a single simulation-snapshot extraction.

```
Required: suite, run_id, snapshot_path. All other fields have documented
defaults matching the Phase A locked spec (Volume 3 Chapter 2).
"""

suite: str
run_id: str
snapshot_path: str
snapshot_time_gyr: Optional[float] = None
centering: str = "potential_min"
primary_temperature: str = "spectroscopic_like"
r_min_kpc: Optional[float] = None
r_max_kpc: Optional[float] = None
n_log_bins: int = 80  # X-COP convention
mu_e: float = MU_E_DEFAULT
physics_flags: dict[str, Any] = field(default_factory=dict)
smooth_control_rule: Optional[str] = None
notes: str = ""
R200_kpc_hint: Optional[float] = None
R500_kpc_hint: Optional[float] = None
z: Optional[float] = None

def __post_init__(self) -> None:
    if self.centering not in _ALLOWED_CENTERING:
        raise ValueError(
            f"centering must be one of {_ALLOWED_CENTERING}; got {self.centering!r}"
        )
    if self.primary_temperature not in _ALLOWED_PRIMARY_T:
        raise ValueError(
            f"primary_temperature must be one of {_ALLOWED_PRIMARY_T}; "
            f"got {self.primary_temperature!r}"
        )
    if self.n_log_bins < 10:
        raise ValueError(f"n_log_bins must be >= 10; got {self.n_log_bins}")
```

# —————————————————————————

# Grid + manual binning (pure NumPy, no yt dependency)

# —————————————————————————

def make_log_grid(
r_min_kpc: float, r_max_kpc: float, n_log_bins: int = 80
) -> np.ndarray:
“”“Return n_log_bins log-spaced bin centers between r_min_kpc and r_max_kpc.

```
Matches the X-COP profile convention (default 80 log-spaced points).
"""
if not (r_min_kpc > 0 and r_max_kpc > r_min_kpc):
    raise ValueError(f"require 0 < r_min < r_max; got {r_min_kpc}, {r_max_kpc}")
return np.geomspace(r_min_kpc, r_max_kpc, n_log_bins)
```

def _bin_edges_from_centers(r_centers_kpc: np.ndarray) -> np.ndarray:
“”“Geometric-midpoint edges from log-spaced centers.

```
Edges have length len(r_centers) + 1. Inner edges are the geometric
midpoints between adjacent centers; the outer edges extrapolate by
half a log-step.
"""
log_r = np.log10(r_centers_kpc)
log_edges = np.empty(len(r_centers_kpc) + 1)
log_edges[1:-1] = 0.5 * (log_r[:-1] + log_r[1:])
log_edges[0] = log_r[0] - 0.5 * (log_r[1] - log_r[0])
log_edges[-1] = log_r[-1] + 0.5 * (log_r[-1] - log_r[-2])
return 10.0 ** log_edges
```

def bin_profiles_from_cells(
r_kpc: np.ndarray,
cell_mass_g: np.ndarray,
cell_density_g_per_cm3: np.ndarray,
cell_temperature_keV: np.ndarray,
r_grid_kpc: np.ndarray,
mu_e: float = MU_E_DEFAULT,
) -> dict[str, np.ndarray]:
“”“Bin per-cell gas data into radial profiles on r_grid_kpc.

```
Pure NumPy. No yt dependency. Each cell is assigned wholly to one bin
based on the cell-center radius. This is correct only when cells are
smaller than bins — for AMR data where cells exceed bin width, use
sim_to_csv_yt_native instead (see Volume 3 Chapter 2 Addendum A).

Inputs are arrays of per-cell quantities (one entry per gas cell or
SPH particle) of identical shape.

Returns dict with arrays of length len(r_grid_kpc):
    ne_cm3            : electron density per shell
    T_sl_keV          : spectroscopic-like temperature (Mazzotta+04)
    T_ew_keV          : emission-weighted temperature (m*rho proxy)
    T_mw_keV          : mass-weighted temperature
    n_cells           : number of cells contributing to each bin
    shell_volume_cm3  : shell volume
    gas_mass_shell_g  : total gas mass in shell
"""
arrs = (r_kpc, cell_mass_g, cell_density_g_per_cm3, cell_temperature_keV)
if len({a.shape for a in arrs}) != 1:
    raise ValueError("per-cell arrays must share the same shape")

edges_kpc = _bin_edges_from_centers(r_grid_kpc)
n_bins = len(r_grid_kpc)

ne = np.zeros(n_bins)
T_sl = np.zeros(n_bins)
T_ew = np.zeros(n_bins)
T_mw = np.zeros(n_bins)
n_cells_per_bin = np.zeros(n_bins, dtype=int)
shell_mass = np.zeros(n_bins)

edges_cm = edges_kpc * CM_PER_KPC
shell_vol_cm3 = (4.0 / 3.0) * np.pi * (edges_cm[1:] ** 3 - edges_cm[:-1] ** 3)

bin_idx = np.digitize(r_kpc, edges_kpc) - 1
valid = (bin_idx >= 0) & (bin_idx < n_bins)

for i in range(n_bins):
    sel = valid & (bin_idx == i)
    if not np.any(sel):
        continue
    m = cell_mass_g[sel]
    rho = cell_density_g_per_cm3[sel]
    T = cell_temperature_keV[sel]

    gas_mass_total = float(m.sum())
    rho_shell = gas_mass_total / shell_vol_cm3[i]
    ne[i] = rho_shell / (mu_e * M_PROTON_G)

    T_mw[i] = float(np.sum(m * T) / np.sum(m))

    # Emission-weighted: w ∝ m*rho (bremsstrahlung-only proxy for hot ICM;
    # full implementation would multiply by Λ(T,Z) cooling function).
    w_ew = m * rho
    T_ew[i] = float(np.sum(w_ew * T) / np.sum(w_ew))

    # Spectroscopic-like (Mazzotta+04): w ∝ rho^2 * T^(-3/4) * dV = m*rho*T^(-3/4)
    w_sl = m * rho * np.power(T, -0.75)
    T_sl[i] = float(np.sum(w_sl * T) / np.sum(w_sl))

    n_cells_per_bin[i] = int(sel.sum())
    shell_mass[i] = gas_mass_total

return {
    "ne_cm3": ne,
    "T_sl_keV": T_sl,
    "T_ew_keV": T_ew,
    "T_mw_keV": T_mw,
    "n_cells": n_cells_per_bin,
    "shell_volume_cm3": shell_vol_cm3,
    "gas_mass_shell_g": shell_mass,
}
```

# —————————————————————————

# CSV + metadata writers

# —————————————————————————

def write_csv_pair(
out_dir: str | Path,
base_name: str,
r_grid_kpc: np.ndarray,
profiles: dict[str, np.ndarray],
primary_T_key: str = “T_sl_keV”,
) -> tuple[Path, Path]:
“”“Write the n_e and T CSVs in the X-COP schema with sim-specific extras.

```
Schema:
    <base>_ne.csv  columns: r_kpc, ne_cm3, n_cells
    <base>_T.csv   columns: r_kpc, T_keV, T_sl_keV, T_ew_keV, T_mw_keV

T_keV is a copy of the primary weighting (default T_sl_keV)
so that cesd.from_csv(T_columns=('r_kpc','T_keV',None))
works unchanged.
"""
if primary_T_key not in profiles:
    raise KeyError(f"primary_T_key {primary_T_key!r} not in profiles")

out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

ne_path = out_dir / f"{base_name}_ne.csv"
T_path = out_dir / f"{base_name}_T.csv"

with open(ne_path, "w") as f:
    f.write("r_kpc,ne_cm3,n_cells\n")
    for r, ne, nc in zip(r_grid_kpc, profiles["ne_cm3"], profiles["n_cells"]):
        f.write(f"{r:.6e},{ne:.6e},{int(nc)}\n")

primary = profiles[primary_T_key]
with open(T_path, "w") as f:
    f.write("r_kpc,T_keV,T_sl_keV,T_ew_keV,T_mw_keV\n")
    for r, t_p, t_sl, t_ew, t_mw in zip(
        r_grid_kpc,
        primary,
        profiles["T_sl_keV"],
        profiles["T_ew_keV"],
        profiles["T_mw_keV"],
    ):
        f.write(f"{r:.6e},{t_p:.6e},{t_sl:.6e},{t_ew:.6e},{t_mw:.6e}\n")

return ne_path, T_path
```

def _config_hash(snapshot: SimSnapshot) -> str:
“”“Stable short hash of the snapshot configuration + loader version.”””
payload = json.dumps(
{“version”: LOADER_VERSION, “snapshot”: asdict(snapshot)},
sort_keys=True,
default=str,
)
return hashlib.sha256(payload.encode()).hexdigest()[:16]

def write_metadata(
out_dir: str | Path,
base_name: str,
snapshot: SimSnapshot,
r_grid_kpc: np.ndarray,
profiles: dict[str, np.ndarray],
extraction_info: Optional[dict[str, Any]] = None,
) -> Path:
“”“Write the JSON metadata sidecar.”””
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
meta_path = out_dir / f”{base_name}_meta.json”

```
populated = profiles["n_cells"] > 0
if populated.any():
    ne_min = float(np.min(profiles["ne_cm3"][populated]))
    T_sl_min = float(np.min(profiles["T_sl_keV"][populated]))
    min_cells = int(profiles["n_cells"][populated].min())
else:
    ne_min = 0.0
    T_sl_min = 0.0
    min_cells = 0

meta = {
    "loader_version": LOADER_VERSION,
    "config_hash": _config_hash(snapshot),
    "snapshot": asdict(snapshot),
    "grid": {
        "n_bins": int(len(r_grid_kpc)),
        "r_min_kpc": float(r_grid_kpc[0]),
        "r_max_kpc": float(r_grid_kpc[-1]),
        "log_spaced": True,
    },
    "profile_summary": {
        "ne_cm3_min": ne_min,
        "ne_cm3_max": float(np.max(profiles["ne_cm3"])),
        "T_sl_keV_min": T_sl_min,
        "T_sl_keV_max": float(np.max(profiles["T_sl_keV"])),
        "n_bins_with_cells": int(populated.sum()),
        "min_cells_per_bin": min_cells,
    },
    "extraction": extraction_info or {},
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2, default=str)

return meta_path
```

# —————————————————————————

# yt helpers (yt-dependent; imported lazily inside functions)

# —————————————————————————

def _find_center_yt(ds: Any, centering: str) -> Any:
“”“Apply the locked centering rule to a yt dataset.

```
For 'potential_min', falls back through ('gas','gravitational_potential')
-> ('flash','gpot') -> max of ('gas','density'). The choice actually used
is recorded in the metadata sidecar via the calling orchestrator.
"""
if centering == "potential_min":
    try:
        _, center = ds.find_min(("gas", "gravitational_potential"))
    except Exception:
        try:
            _, center = ds.find_min(("flash", "gpot"))
        except Exception:
            _, center = ds.find_max(("gas", "density"))
elif centering == "density_peak":
    _, center = ds.find_max(("gas", "density"))
else:
    raise NotImplementedError(
        f"centering={centering!r} not implemented in v{LOADER_VERSION}"
    )
return center
```

def extract_raw_gas_data_via_yt(snapshot: SimSnapshot) -> dict[str, Any]:
“”“Read a snapshot via yt and return per-cell arrays.

```
Low-level utility, retained for non-AMR / non-yt-create_profile workflows
or for downstream code that wants per-cell data directly. For AMR data
that goes straight to CSV, prefer sim_to_csv_yt_native.

Returns a dict with:
    r_kpc                   : radial distance from chosen center, kpc
    cell_mass_g             : per-cell gas mass, g
    cell_density_g_per_cm3  : per-cell mass density, g/cm^3
    cell_temperature_keV    : per-cell temperature, keV
    center                  : (x, y, z) in code units
"""
try:
    import yt  # type: ignore
except ImportError as e:
    raise ImportError(
        "yt is required for extract_raw_gas_data_via_yt. "
        "Install with: pip install yt"
    ) from e

ds = yt.load(snapshot.snapshot_path)
center = _find_center_yt(ds, snapshot.centering)

r_max_kpc = snapshot.r_max_kpc or 2000.0
sp = ds.sphere(center, (r_max_kpc, "kpc"))

# NOTE: ('index','radius') is the correct field on most yt datasets,
# including FLASH. ('gas','radius') was incorrect and was the cause of
# the v0.1 → v0.2 patch (Volume 3 Chapter 2 working notes).
r_cm = np.asarray(sp[("index", "radius")].in_cgs())
r_kpc = r_cm / CM_PER_KPC

mass_g = np.asarray(sp[("gas", "mass")].in_cgs())
rho = np.asarray(sp[("gas", "density")].in_cgs())
T_K = np.asarray(sp[("gas", "temperature")].in_cgs())
T_keV = T_K * KEV_PER_K

return {
    "r_kpc": r_kpc,
    "cell_mass_g": mass_g,
    "cell_density_g_per_cm3": rho,
    "cell_temperature_keV": T_keV,
    "center": tuple(float(c) for c in np.asarray(center)),
}
```

# —————————————————————————

# Top-level orchestrators

# —————————————————————————

def _resolve_base_name(snapshot: SimSnapshot, base_name: Optional[str]) -> str:
if base_name is not None:
return base_name
if snapshot.snapshot_time_gyr is not None:
t_str = f”_t{snapshot.snapshot_time_gyr:.2f}”.replace(”.”, “p”)
else:
t_str = “”
return f”{snapshot.run_id}{t_str}”

def sim_to_csv(
snapshot: SimSnapshot,
out_dir: str | Path,
*,
cell_data: dict[str, Any],
base_name: Optional[str] = None,
) -> dict[str, Path]:
“”“Bin pre-extracted per-cell gas arrays into the X-COP CSV schema.

```
Pure-NumPy path via bin_profiles_from_cells. No yt dependency.

Use this for synthetic input or for non-yt sources where you have already
extracted cell-level data. **Not recommended for AMR data** where cell
sizes can exceed bin widths — use sim_to_csv_yt_native instead (see
Volume 3 Chapter 2 Addendum A).

Parameters
----------
snapshot : SimSnapshot
    Configuration. r_min_kpc / r_max_kpc on the snapshot, if None,
    default to data-driven values from cell_data.
out_dir : path
    Output directory; created if it doesn't exist.
cell_data : dict
    Per-cell arrays as returned by extract_raw_gas_data_via_yt or by
    a synthetic generator. Keys required:
    r_kpc, cell_mass_g, cell_density_g_per_cm3, cell_temperature_keV
    Optional: 'center' (recorded in metadata).
base_name : str, optional
    Output filename prefix. Auto-derived from run_id and snapshot_time
    if not provided.

Returns
-------
dict with keys 'ne_csv', 'T_csv', 'meta_json' mapping to Path objects.
"""
r_kpc = np.asarray(cell_data["r_kpc"])
positive_r = r_kpc[r_kpc > 0]
if positive_r.size == 0:
    raise ValueError("no cells with r > 0 found in cell_data")

r_min = (
    snapshot.r_min_kpc
    if snapshot.r_min_kpc is not None
    else max(5.0, float(positive_r.min()))
)
r_max = (
    snapshot.r_max_kpc
    if snapshot.r_max_kpc is not None
    else float(np.percentile(positive_r, 99.9))
)
if r_max <= r_min:
    raise ValueError(f"resolved r_max ({r_max}) <= r_min ({r_min})")

r_grid = make_log_grid(r_min, r_max, snapshot.n_log_bins)

profiles = bin_profiles_from_cells(
    r_kpc=r_kpc,
    cell_mass_g=np.asarray(cell_data["cell_mass_g"]),
    cell_density_g_per_cm3=np.asarray(cell_data["cell_density_g_per_cm3"]),
    cell_temperature_keV=np.asarray(cell_data["cell_temperature_keV"]),
    r_grid_kpc=r_grid,
    mu_e=snapshot.mu_e,
)

primary_T_key = _PRIMARY_T_KEY_MAP[snapshot.primary_temperature]
base_name = _resolve_base_name(snapshot, base_name)
ne_path, T_path = write_csv_pair(out_dir, base_name, r_grid, profiles, primary_T_key)

extraction_info = {
    "binning_method": "bin_profiles_from_cells (manual, pure NumPy)",
    "n_cells_total": int(r_kpc.size),
    "n_cells_in_range": int(((r_kpc >= r_min) & (r_kpc <= r_max)).sum()),
    "center_used": cell_data.get("center"),
    "r_min_kpc_resolved": float(r_min),
    "r_max_kpc_resolved": float(r_max),
    "primary_T_key": primary_T_key,
}
meta_path = write_metadata(out_dir, base_name, snapshot, r_grid, profiles, extraction_info)

return {"ne_csv": ne_path, "T_csv": T_path, "meta_json": meta_path}
```

def sim_to_csv_yt_native(
snapshot: SimSnapshot,
out_dir: str | Path,
*,
base_name: Optional[str] = None,
) -> dict[str, Path]:
“”“Read a snapshot via yt and bin it with yt.create_profile.

```
The validated AMR path (Phase A checkpoint, Volume 3 Chapter 2
Addendum A). yt's create_profile handles cell-to-bin geometry correctly
via internal volume weighting: every cell contributes to every bin it
overlaps, weighted by intersection volume. This is the right algorithm
for AMR data on radial grids and is what yt was built to do.

Five yt.create_profile calls produce the per-bin quantities:
    1. volume-weighted ('gas','density')           -> n_e (with mu_e)
    2. mass-weighted ('gas','temperature')         -> T_mw
    3. m·rho-weighted ('gas','temperature')        -> T_ew
    4. m·rho·T^(-3/4)-weighted ('gas','temp')      -> T_sl (Mazzotta+04)
    5. unweighted ('index','ones')                 -> n_cells per bin

Parameters
----------
snapshot : SimSnapshot
    Configuration.
out_dir : path
    Output directory; created if it doesn't exist.
base_name : str, optional
    Output filename prefix. Auto-derived if not provided.

Returns
-------
dict with keys 'ne_csv', 'T_csv', 'meta_json' mapping to Path objects.

Raises
------
ImportError
    If yt is not installed in the current environment.
"""
try:
    import yt  # type: ignore
except ImportError as e:
    raise ImportError(
        "yt is required for sim_to_csv_yt_native. Install with: pip install yt"
    ) from e

ds = yt.load(snapshot.snapshot_path)
center = _find_center_yt(ds, snapshot.centering)

r_min = snapshot.r_min_kpc if snapshot.r_min_kpc is not None else 50.0
r_max = snapshot.r_max_kpc if snapshot.r_max_kpc is not None else 800.0
if r_max <= r_min:
    raise ValueError(f"r_max ({r_max}) <= r_min ({r_min})")
n_bins = snapshot.n_log_bins

sp = ds.sphere(center, (r_max, "kpc"))

# Register custom weight fields for T_ew and T_sl. force_override=True so
# repeated calls don't error. Units declared as g**2/cm**3 — yt does not
# require dimensional consistency on weight fields used purely as weights
# (the units cancel in the weighted-mean numerator and denominator).
def _ew_weight(field, data):
    return data[("gas", "mass")] * data[("gas", "density")]

def _sl_weight(field, data):
    T = data[("gas", "temperature")].in_units("K").to_value()
    return data[("gas", "mass")] * data[("gas", "density")] * (T ** -0.75)

if ("gas", "ew_weight") not in ds.derived_field_list:
    ds.add_field(
        ("gas", "ew_weight"),
        function=_ew_weight,
        sampling_type="cell",
        units="g**2/cm**3",
        force_override=True,
    )
if ("gas", "sl_weight") not in ds.derived_field_list:
    ds.add_field(
        ("gas", "sl_weight"),
        function=_sl_weight,
        sampling_type="cell",
        units="g**2/cm**3",
        force_override=True,
    )

profile_common = dict(
    n_bins=n_bins,
    units={("index", "radius"): "kpc"},
    extrema={("index", "radius"): (r_min, r_max)},
    logs={("index", "radius"): True},
)

p_rho = yt.create_profile(
    sp,
    [("index", "radius")],
    [("gas", "density")],
    weight_field=("index", "cell_volume"),
    **profile_common,
)
p_Tmw = yt.create_profile(
    sp,
    [("index", "radius")],
    [("gas", "temperature")],
    weight_field=("gas", "mass"),
    **profile_common,
)
p_Tew = yt.create_profile(
    sp,
    [("index", "radius")],
    [("gas", "temperature")],
    weight_field=("gas", "ew_weight"),
    **profile_common,
)
p_Tsl = yt.create_profile(
    sp,
    [("index", "radius")],
    [("gas", "temperature")],
    weight_field=("gas", "sl_weight"),
    **profile_common,
)
p_count = yt.create_profile(
    sp,
    [("index", "radius")],
    [("index", "ones")],
    weight_field=None,  # unweighted: sum of "ones" per bin == cell count
    **profile_common,
)

r_grid = np.asarray(p_rho.x.in_units("kpc"))
rho_g_per_cm3 = np.asarray(p_rho[("gas", "density")].in_units("g/cm**3"))
T_mw_K = np.asarray(p_Tmw[("gas", "temperature")].in_units("K"))
T_ew_K = np.asarray(p_Tew[("gas", "temperature")].in_units("K"))
T_sl_K = np.asarray(p_Tsl[("gas", "temperature")].in_units("K"))
n_cells = np.asarray(p_count[("index", "ones")]).astype(int)

profiles = {
    "ne_cm3": rho_g_per_cm3 / (snapshot.mu_e * M_PROTON_G),
    "T_sl_keV": T_sl_K * KEV_PER_K,
    "T_ew_keV": T_ew_K * KEV_PER_K,
    "T_mw_keV": T_mw_K * KEV_PER_K,
    "n_cells": n_cells,
    # shell_volume_cm3 / gas_mass_shell_g not exposed by yt.create_profile;
    # filled with zeros for schema consistency with the manual-binning path.
    "shell_volume_cm3": np.zeros_like(r_grid),
    "gas_mass_shell_g": np.zeros_like(r_grid),
}

primary_T_key = _PRIMARY_T_KEY_MAP[snapshot.primary_temperature]
base_name = _resolve_base_name(snapshot, base_name)
ne_path, T_path = write_csv_pair(out_dir, base_name, r_grid, profiles, primary_T_key)

extraction_info = {
    "binning_method": "yt.create_profile (volume-weighted, AMR-correct)",
    "yt_version": getattr(yt, "__version__", "unknown"),
    "center_used": tuple(float(c) for c in np.asarray(center)),
    "centering_choice": snapshot.centering,
    "r_min_kpc_resolved": float(r_min),
    "r_max_kpc_resolved": float(r_max),
    "primary_T_key": primary_T_key,
}
meta_path = write_metadata(out_dir, base_name, snapshot, r_grid, profiles, extraction_info)

return {"ne_csv": ne_path, "T_csv": T_path, "meta_json": meta_path}
```

**all** = [
“LOADER_VERSION”,
“M_PROTON_G”,
“CM_PER_KPC”,
“KEV_PER_K”,
“MU_E_DEFAULT”,
“SimSnapshot”,
“make_log_grid”,
“bin_profiles_from_cells”,
“write_csv_pair”,
“write_metadata”,
“extract_raw_gas_data_via_yt”,
“sim_to_csv”,
“sim_to_csv_yt_native”,
]
