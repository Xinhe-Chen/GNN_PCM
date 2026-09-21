"""
Mechanism-based LMP recovery from a DC dispatch, validated against Prescient.

Rather than regressing locational marginal prices with a second network, this
reconstructs them from the DC-OPF dual. For a linearized dispatch the price at
bus n decomposes exactly as

    LMP_n = lambda  -  sum_{l in B} PTDF[l, n] * nu_l
            \______/    \_________________________/
            energy              congestion

where B is the set of binding branches and nu_l is the (signed) shadow price of
branch l. Lagrangian stationarity for any *marginal* unit i -- one dispatched
strictly inside [PMin, PMax], so its own capacity duals vanish -- gives

    C'_i = lambda - sum_{l in B} PTDF[l, bus(i)] * nu_l

which is a small linear system in (lambda, nu). With a median of one binding
branch and a handful of marginal units per hour, it solves in microseconds, and
the resulting prices are consistent by construction with the dispatch and the
flows the GNN already predicts.

Why this beats a learned LMP head
---------------------------------
Prescient's LMP distribution on the base case spans -$8.85 to $653/MWh with
26.6% of entries exactly zero and, in 88% of hours, no spatial variation at
all. A smooth regressor fights all of that. The dual does not: it reproduces
the piecewise-constant structure because the structure *is* the active set.

Four empirical facts about this PCM run motivate the special-case handling
below; each is measured by `--report-diagnostics`:

  1. With no binding branch, LMP is spatially uniform in 100.0% of hours.
     So congestion is the only source of spatial spread, exactly as the dual
     predicts.
  2. lambda == 0 coincides with renewable curtailment > 0 in 99.9% of the
     1,827 zero-price hours: a zero-marginal-cost renewable is on the margin,
     so no thermal unit sets the price.
  3. lambda equals *some* up-capable unit's segment marginal cost to within
     $0.01 in 89.1% of uncongested hours -- but at median rank 0.50 of ~12
     candidates, so no ordering heuristic finds it. This is what makes the
     learnable part a classification rather than a regression (see
     `candidate_lambdas` and the "oracle" method below).
  4. The 10.9% of hours where lambda matches no unit are reserve-binding:
     98.2% of them carry a nonzero reserve price (mean $61.24), against 6.9%
     (mean $0.09) elsewhere. Prescient co-optimizes six reserve products, and
     their duals enter a price this energy-only stationarity condition cannot
     see.

Cost curves
-----------
RTS-GMLC specifies piecewise-linear incremental heat rates, not a quadratic.
Segment k spans [Output_pct_{k-1}, Output_pct_k] * PMax with marginal cost
HR_incr_k * Fuel Price / 1000 + VOM. This was verified against Prescient's own
accounting by finite-differencing the `Unit Cost` column against `Dispatch`:
the empirical dC/dP lands on these segment values exactly.

Methods
-------
    ls      Pure least squares over the marginal units. Fully analytic, no
            learning, no labels -- the honest baseline for the mechanism.
    oracle  lambda is chosen from the discrete candidate set to best match the
            observed price; nu still comes from least squares. This is NOT a
            usable predictor -- it reads the answer. It measures the ceiling a
            perfect marginal-unit classifier would reach, which is the number
            worth knowing before building one.

Usage
-----
    python lmp_derivation.py                      # both methods, per-bus MAE
    python lmp_derivation.py --method ls
    python lmp_derivation.py --report-diagnostics # the four facts above
    python lmp_derivation.py --csv-out lmp_mae.csv

Solver settings read from the PCM configuration
-----------------------------------------------
`PRICE_THRESHOLD` below is Prescient's own `price_threshold` (500 $/MWh in
src/Prescient_synthetic_whole_year/pcm_run_using_synthetic.py). It is
load-bearing: using it for the load-shedding hours takes their MAE from
$494.18 to $3.57, and the overall figure from $14.996 to $6.115. Rerunning the
PCM with a different `shortfall` requires changing it here to match.

`NU_MAX` is NOT a PCM setting -- that run leaves congestion prices uncapped --
and is off by default. See its comment for why it should not be tuned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import lsq_linear

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

import data_prep

THERMAL_UNIT_TYPES = data_prep.THERMAL_UNIT_TYPES

#: A branch counts as binding at this fraction of its continuous rating.
#: Prescient enforces limits softly, so realized flows reach 148% of rating;
#: anything at or above this level is treated as constrained.
BINDING_THRESHOLD = 0.99

#: Tolerance (MW) for calling a unit interior, or a dispatch level a segment
#: breakpoint. Prescient writes dispatch to ~1e-6 MW, so 0.5 MW is loose enough
#: to absorb rounding without swallowing genuinely part-loaded units.
MW_TOL = 0.5

#: Price agreement tolerance ($/MWh) for the "lambda equals a candidate" test.
PRICE_TOL = 0.01

#: Ridge weight on the congestion prices, applied to unit-norm-scaled columns.
#: Resolves the unidentified directions of a near-singular dual system toward
#: zero congestion rather than toward huge offsetting prices.
RIDGE = 1e-3

#: Optional hard cap ($/MWh) on a single branch's congestion price. None is
#: the deliberate default because it matches the PCM configuration:
#: `pcm_run_using_synthetic.py` sets transmission_price_threshold=None, so
#: Prescient does NOT cap congestion prices and no value here corresponds to a
#: solver setting. The guard against an ill-conditioned solve is the ridge term
#: in `solve_hour`, not this.
#:
#: Tightening it does improve the fit, monotonically and mildly -- ALL MAE runs
#: $6.115 (uncapped), $6.082 ($2000), $6.058 ($1000), $6.018 ($500), $5.997
#: ($200) -- but that is a symptom, not a feature. It means the solve
#: over-attributes price differences to congestion when the marginal-unit set
#: is misidentified (see the reserve-product gap in fact 4), and clamping nu
#: happens to offset that bias. Tuning this against the observed prices would
#: be fitting a regularizer to the validation target for ~$0.12/MWh; fix the
#: reserve term instead.
NU_MAX = None

#: Prescient's `price_threshold`, which caps the ENERGY component of the price.
#: Read straight from the run configuration
#: (src/Prescient_synthetic_whole_year/pcm_run_using_synthetic.py: shortfall = 500),
#: and confirmed in the output: the 159 load-shedding hours price at a median of
#: exactly $500.00, and 145 hours have a system-mean LMP of exactly $500.
#:
#: It bounds lambda, not the LMP: congestion rides on top, so realized LMPs
#: reach $653/MWh (0.10% of entries exceed $500). Change this if you rerun the
#: PCM with a different `shortfall`.
PRICE_THRESHOLD = 500.0


# ---------------------------------------------------------------------------
# Cost curves
# ---------------------------------------------------------------------------

def thermal_cost_curves(gen_csv: Path = data_prep.GEN_CSV):
    """Piecewise-linear incremental cost curve for every thermal unit.

    Returns dict: GEN UID -> {bus_id, pmin, pmax, segments}, where `segments`
    is a list of (lo_mw, hi_mw, marginal_cost) tuples in increasing output
    order. Output_pct_0 is the unit's PMin as a fraction of PMax, so the first
    segment starts at PMin rather than at zero.

    Units whose heat-rate columns are incomplete get an empty segment list and
    are skipped as lambda candidates rather than silently priced at zero.
    """
    gen = pd.read_csv(gen_csv).set_index("GEN UID")
    curves = {}

    for uid in gen.index:
        row = gen.loc[uid]
        if row["Unit Type"] not in THERMAL_UNIT_TYPES:
            continue

        pmax = float(row["PMax MW"])
        fuel = float(row["Fuel Price $/MMBTU"])
        vom = float(row["VOM"]) if not pd.isna(row["VOM"]) else 0.0
        pct = [row.get(f"Output_pct_{k}") for k in range(5)]
        incr = [row.get(f"HR_incr_{k}") for k in range(1, 5)]

        segments = []
        for k in range(4):
            if k + 1 >= len(pct) or pd.isna(pct[k]) or pd.isna(pct[k + 1]) or pd.isna(incr[k]):
                continue
            lo, hi = float(pct[k]) * pmax, float(pct[k + 1]) * pmax
            segments.append((lo, hi, float(incr[k]) * fuel / 1000.0 + vom))

        curves[uid] = {
            "bus_id": int(row["Bus ID"]),
            "pmin": float(row["PMin MW"]),
            "pmax": pmax,
            "segments": segments,
        }
    return curves


def marginal_cost_at(curve: dict, output_mw: float):
    """Marginal cost of the segment containing `output_mw`, or None if unknown.

    At an interior breakpoint the derivative is set-valued; this returns the
    upper segment's cost, and `candidate_lambdas` adds the lower one so the
    oracle can pick either.
    """
    segments = curve["segments"]
    if not segments:
        return None
    for lo, hi, cost in segments:
        if lo - 1e-6 <= output_mw <= hi + 1e-6:
            return cost
    return segments[-1][2] if output_mw > segments[-1][1] else segments[0][2]


def candidate_lambdas(curve: dict, output_mw: float):
    """Segment costs that could set lambda if this unit were marginal.

    A unit part-loaded at a breakpoint can price at either adjacent segment
    depending on the direction of the next MW, so both are returned. This is
    the label space for the marginal-unit classifier the module docstring
    describes.
    """
    out = []
    for lo, hi, cost in curve["segments"]:
        if lo - MW_TOL <= output_mw <= hi + MW_TOL:
            out.append(cost)
    if not out:
        cost = marginal_cost_at(curve, output_mw)
        if cost is not None:
            out.append(cost)
    return out


# ---------------------------------------------------------------------------
# Hourly PCM state
# ---------------------------------------------------------------------------

def _hour_index(df: pd.DataFrame):
    """Dense chronological hour index from (Date, Hour), shared across files."""
    keys = df[["Date", "Hour"]].drop_duplicates().sort_values(["Date", "Hour"]).reset_index(drop=True)
    keys["_t"] = np.arange(len(keys))
    return df.merge(keys, on=["Date", "Hour"], how="left"), keys


def load_pcm_state(run_dir: Path, curves: dict, bus_ids: np.ndarray, line_ids: np.ndarray, line_ratings: np.ndarray):
    """Assemble everything the dual solve needs, hour by hour.

    Returns a dict of arrays indexed by hour (and unit / bus / branch where
    relevant). `dispatch`, `interior`, `up_capable` and `unit_mc` are
    unit-level because the stationarity condition is a per-unit statement --
    aggregating to the bus would destroy the merit-order information the dual
    is built on.
    """
    run_dir = Path(run_dir)

    thermal = pd.read_csv(run_dir / "thermal_detail.csv")
    thermal = thermal[thermal["Generator"].isin(curves)].copy()
    thermal, keys = _hour_index(thermal)
    n_hours = len(keys)

    uids = np.array(sorted(curves))
    uid_pos = {u: i for i, u in enumerate(uids)}
    thermal["_u"] = thermal["Generator"].map(uid_pos)

    dispatch = np.zeros((n_hours, len(uids)))
    on = np.zeros((n_hours, len(uids)), dtype=bool)
    dispatch[thermal["_t"], thermal["_u"]] = thermal["Dispatch"].to_numpy()
    on[thermal["_t"], thermal["_u"]] = thermal["Unit State"].astype(bool).to_numpy()

    pmin = np.array([curves[u]["pmin"] for u in uids])
    pmax = np.array([curves[u]["pmax"] for u in uids])
    unit_bus = np.array([curves[u]["bus_id"] for u in uids])

    interior = on & (dispatch > pmin + MW_TOL) & (dispatch < pmax - MW_TOL)
    up_capable = on & (dispatch < pmax - MW_TOL)

    # Branch flows, excluding the DC tie (no PTDF relationship).
    lines = pd.read_csv(run_dir / "line_detail.csv")
    lines = lines[lines["Line"] != data_prep.DC_BRANCH_UID].copy()
    lines, _ = _hour_index(lines)
    flow = (
        lines.pivot_table(index="_t", columns="Line", values="Flow", aggfunc="sum")
        .reindex(index=range(n_hours))
        .reindex(columns=list(line_ids))
        .to_numpy(dtype=np.float64)
    )
    utilization = np.abs(flow) / line_ratings[None, :]

    # Observed prices (the validation target).
    bus_detail = pd.read_csv(run_dir / "bus_detail.csv", usecols=["Date", "Hour", "Bus", "LMP"])
    bus_detail, _ = _hour_index(bus_detail)
    bus_meta = pd.read_csv(data_prep.BUS_CSV)
    name_to_id = dict(zip(bus_meta["Bus Name"], bus_meta["Bus ID"]))
    bus_detail["bus_id"] = bus_detail["Bus"].map(name_to_id)
    lmp = (
        bus_detail.pivot_table(index="_t", columns="bus_id", values="LMP", aggfunc="mean")
        .reindex(index=range(n_hours))
        .reindex(columns=[int(b) for b in bus_ids])
        .to_numpy(dtype=np.float64)
    )

    renew = pd.read_csv(run_dir / "renewables_detail.csv", usecols=["Date", "Hour", "Curtailment"])
    renew, _ = _hour_index(renew)
    curtailment = renew.groupby("_t")["Curtailment"].sum().reindex(range(n_hours)).fillna(0.0).to_numpy()

    reserves = pd.read_csv(run_dir / "reserves_detail.csv", usecols=["Date", "Hour", "Reserve", "Scope", "Shortfall", "Price"])
    reserves, _ = _hour_index(reserves)
    reserve_price = (
        reserves.groupby("_t")["Price"].max().reindex(range(n_hours)).fillna(0.0).to_numpy()
    )
    # A *shortfall* is the sharp cut, not merely a nonzero price: when a reserve
    # product cannot be met, its dual sits at the shortfall penalty and adds
    # directly to the energy price, which the energy-only stationarity condition
    # in `solve_hour` cannot see. These 840 hours carry essentially all of the
    # method's bias -- see the module docstring.
    reserve_shortfall = (
        reserves.groupby("_t")["Shortfall"].max().reindex(range(n_hours)).fillna(0.0).to_numpy()
    )
    short = reserves[reserves["Shortfall"] > 1e-6]
    reserve_short_price = (
        short.groupby("_t")["Price"].max().reindex(range(n_hours)).fillna(0.0).to_numpy()
        if len(short)
        else np.zeros(n_hours)
    )

    hourly = pd.read_csv(run_dir / "hourly_summary.csv")
    hourly = keys.merge(hourly, on=["Date", "Hour"], how="left")
    load_shed = hourly["LoadShedding"].fillna(0.0).to_numpy()

    return {
        "keys": keys,
        "uids": uids,
        "unit_bus": unit_bus,
        "dispatch": dispatch,
        "interior": interior,
        "up_capable": up_capable,
        "flow": flow,
        "utilization": utilization,
        "lmp": lmp,
        "curtailment": curtailment,
        "reserve_price": reserve_price,
        "reserve_shortfall": reserve_shortfall,
        "reserve_short_price": reserve_short_price,
        "load_shed": load_shed,
    }


# ---------------------------------------------------------------------------
# The dual solve
# ---------------------------------------------------------------------------

def solve_hour(
    mc: np.ndarray,
    ptdf_cols: np.ndarray,
    binding_signs: np.ndarray,
    lambda_fixed: float | None = None,
    ridge: float = RIDGE,
    nu_max: float | None = NU_MAX,
    price_threshold: float | None = PRICE_THRESHOLD,
):
    """Recover (lambda, nu) from the marginal units' stationarity conditions.

    Parameters
    ----------
    mc : [m]
        Marginal cost of each marginal unit.
    ptdf_cols : [m, b]
        PTDF of each binding branch with respect to each marginal unit's bus.
    binding_signs : [b]
        Sign of the flow on each binding branch. The dual of an upper limit is
        non-negative and of a lower limit non-positive, so nu is sign-
        constrained; substituting w_l = sign_l * nu_l turns that into a simple
        non-negativity bound.
    lambda_fixed
        Pin the energy price instead of solving for it. Used when renewable
        curtailment forces lambda = 0: a zero-marginal-cost resource is on the
        margin, so no thermal unit sets the price and the thermal stationarity
        conditions only inform the congestion terms.

    Numerical conditioning
    ----------------------
    This system is often close to singular, and handling that is not optional.
    If every marginal unit happens to sit at nearly the same PTDF value with
    respect to a binding branch, that branch's column is almost collinear with
    the lambda column, so nu is unidentified: an unconstrained solve then
    returns congestion prices of order 1e14 with a tiny residual, and the
    reconstructed LMPs are garbage. Three defences, all necessary:

      * columns are scaled to unit norm, so ridge shrinkage is applied equally
        per direction rather than being dominated by whichever branch happens
        to have the largest PTDF entries;
      * a ridge term prefers the smallest congestion prices that explain the
        observed marginal costs, which resolves the unidentified directions
        toward zero instead of toward infinity;
      * nu is optionally bounded by `nu_max`. This is off by default,
        matching the PCM's transmission_price_threshold=None; on this run the
        first two defences alone are sufficient.

    Returns
    -------
    (lambda, nu [b]) or (None, None) when there is nothing to solve.
    """
    n_binding = len(binding_signs)

    if lambda_fixed is None and len(mc) == 0:
        return None, None
    if n_binding == 0:
        lam = float(np.median(mc)) if lambda_fixed is None else lambda_fixed
        if price_threshold:
            lam = min(lam, price_threshold)
        return lam, np.zeros(0)
    if len(mc) == 0:
        # Nothing constrains the congestion terms; ridge would drive them to 0.
        return lambda_fixed, np.zeros(n_binding)

    # Columns for w = sign * nu, so every bound becomes [0, nu_max].
    a_nu = -ptdf_cols * binding_signs[None, :]
    scale = np.linalg.norm(a_nu, axis=0)
    scale[scale < 1e-9] = 1.0
    a_scaled = a_nu / scale[None, :]

    if lambda_fixed is None:
        design = np.concatenate([np.ones((len(mc), 1)), a_scaled], axis=1)
        target = mc.astype(np.float64)
        lower = np.concatenate([[-np.inf], np.zeros(n_binding)])
        cap = np.inf if nu_max is None else nu_max
        upper = np.concatenate([[price_threshold if price_threshold else np.inf], cap * scale])
        # Ridge rows touch the nu columns only -- lambda must not be shrunk
        # toward zero, it is a price level, not a deviation.
        reg = np.concatenate([np.zeros((n_binding, 1)), np.sqrt(ridge) * np.eye(n_binding)], axis=1)
    else:
        design = a_scaled
        target = (mc - lambda_fixed).astype(np.float64)
        lower = np.zeros(n_binding)
        upper = (np.inf if nu_max is None else nu_max) * scale
        reg = np.sqrt(ridge) * np.eye(n_binding)

    design_reg = np.concatenate([design, reg], axis=0)
    target_reg = np.concatenate([target, np.zeros(n_binding)])

    if _HAVE_SCIPY:
        sol = lsq_linear(design_reg, target_reg, bounds=(lower, upper)).x
    else:
        sol, *_ = np.linalg.lstsq(design_reg, target_reg, rcond=None)
        sol = np.clip(sol, lower, upper)

    if lambda_fixed is None:
        return float(sol[0]), (sol[1:] / scale) * binding_signs
    return lambda_fixed, (sol / scale) * binding_signs


def derive_lmp(state: dict, ptdf: np.ndarray, curves: dict, bus_index: dict, method: str = "ls", binding_threshold: float = BINDING_THRESHOLD, ridge: float = RIDGE, price_threshold: float | None = PRICE_THRESHOLD, nu_max: float | None = NU_MAX):
    """Reconstruct LMP at every bus for every hour.

    Special cases, applied in this order:

      load shedding   The price is set by the value-of-lost-load penalty, not
                      by any generator's cost, so lambda is pinned to
                      `price_threshold` (Prescient's own shortfall setting).
                      These hours are also reported as their own regime, since
                      the penalty is a solver setting rather than physics.
      curtailment>0   With no branch binding, lambda is pinned to 0: a
                      zero-marginal-cost renewable is on the margin (fact 2).
                      With a branch binding, curtailment may be local rather
                      than systemic, so lambda is solved for normally.
      no marginal     With no interior unit there is no stationarity condition
                      to solve; lambda falls back to the cheapest up-capable
                      unit's segment cost, or 0 if none exists.

    Returns
    -------
    lmp_hat [T, N], plus per-hour diagnostics (lambda, n_binding, method flags).
    """
    if method not in ("ls", "oracle"):
        raise ValueError(f"method must be 'ls' or 'oracle', got {method!r}")

    uids, unit_bus = state["uids"], state["unit_bus"]
    dispatch, interior, up_capable = state["dispatch"], state["interior"], state["up_capable"]
    utilization, flow = state["utilization"], state["flow"]
    n_hours, n_nodes = len(state["keys"]), ptdf.shape[1]

    unit_node = np.array([bus_index[int(b)] for b in unit_bus])
    lmp_hat = np.zeros((n_hours, n_nodes))
    lam_out = np.zeros(n_hours)
    n_binding_out = np.zeros(n_hours, dtype=int)
    used_oracle = np.zeros(n_hours, dtype=bool)

    for t in range(n_hours):
        binding = np.where(utilization[t] >= binding_threshold)[0]
        signs = np.sign(flow[t, binding])
        signs[signs == 0] = 1.0

        # Marginal units and their segment costs, kept strictly in step: a unit
        # with no usable cost curve is dropped from both arrays, never one.
        marg, mc = [], []
        for i in np.where(interior[t])[0]:
            cost = marginal_cost_at(curves[uids[i]], dispatch[t, i])
            if cost is not None:
                marg.append(i)
                mc.append(cost)
        marg = np.array(marg, dtype=int)
        mc = np.array(mc, dtype=np.float64)

        if price_threshold is not None and state["load_shed"][t] > 0:
            # Load was shed, so the price is set by Prescient's price_threshold
            # rather than by any generator's cost curve.
            lambda_fixed = price_threshold
        elif state["curtailment"][t] > 0 and len(binding) == 0:
            # A curtailed zero-cost renewable is on the margin only when the
            # curtailment is a system-wide surplus. If a branch is binding the
            # renewable may instead be backed down by local congestion, and a
            # thermal unit still sets lambda -- 912 hours in the base case have
            # curtailment > 0 with lambda ~ $20.69, and every curtailment
            # threshold tried tops out at 0.80 precision, so the binding-set
            # test is what actually separates the two cases.
            lambda_fixed = 0.0
        else:
            lambda_fixed = None

        if method == "oracle" and lambda_fixed is None:
            # Pick lambda from the discrete candidate set -- the ceiling for a
            # marginal-unit classifier. Reads the observed price, so this is a
            # bound, not a predictor.
            cands = set()
            for i in np.where(up_capable[t])[0]:
                cands.update(candidate_lambdas(curves[uids[i]], dispatch[t, i]))
            observed_row = state["lmp"][t]
            if cands and np.isfinite(observed_row).any():
                observed = np.nanmean(observed_row) if len(binding) == 0 else np.nanmedian(observed_row)
                cands = np.array(sorted(cands))
                lambda_fixed = float(cands[np.argmin(np.abs(cands - observed))])
                used_oracle[t] = True

        if lambda_fixed is None and len(mc) == 0:
            up = np.where(up_capable[t])[0]
            costs = [marginal_cost_at(curves[uids[i]], dispatch[t, i]) for i in up]
            costs = [c for c in costs if c is not None]
            lambda_fixed = float(min(costs)) if costs else 0.0

        lam, nu = solve_hour(mc, ptdf[np.ix_(binding, unit_node[marg])].T, signs, lambda_fixed,
                             ridge=ridge, nu_max=nu_max, price_threshold=price_threshold)
        if lam is None:
            lam, nu = 0.0, np.zeros(len(binding))

        prices = np.full(n_nodes, lam)
        if len(binding):
            prices = prices - ptdf[binding, :].T @ nu

        lmp_hat[t] = prices
        lam_out[t] = lam
        n_binding_out[t] = len(binding)

    return {
        "lmp_hat": lmp_hat,
        "lambda": lam_out,
        "n_binding": n_binding_out,
        "used_oracle": used_oracle,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def per_bus_mae(lmp_hat: np.ndarray, lmp_true: np.ndarray, bus_ids: np.ndarray, mask: np.ndarray | None = None):
    """MAE in $/MWh at every bus, with the mean observed price for scale."""
    valid = np.isfinite(lmp_true)
    if mask is not None:
        valid = valid & mask[:, None]

    rows = []
    for j, bus in enumerate(bus_ids):
        v = valid[:, j]
        if not v.any():
            rows.append({"bus_id": int(bus), "n_hours": 0, "mae": np.nan, "mean_lmp": np.nan, "max_abs_err": np.nan})
            continue
        err = lmp_hat[v, j] - lmp_true[v, j]
        rows.append(
            {
                "bus_id": int(bus),
                "n_hours": int(v.sum()),
                "mae": float(np.abs(err).mean()),
                "median_ae": float(np.median(np.abs(err))),
                "mean_lmp": float(lmp_true[v, j].mean()),
                "max_abs_err": float(np.abs(err).max()),
                "bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows)


def stratified_summary(lmp_hat: np.ndarray, state: dict, result: dict):
    """Overall MAE, broken out by the regimes that behave differently."""
    lmp_true = state["lmp"]
    valid = np.isfinite(lmp_true)
    err = np.abs(lmp_hat - lmp_true)

    congested = result["n_binding"] > 0
    curtailed = state["curtailment"] > 0
    reserve = state["reserve_price"] > 1e-6
    shortfall = state.get("reserve_shortfall", np.zeros(len(lmp_true))) > 1e-6
    shed = state["load_shed"] > 0

    def stat(hour_mask, label):
        m = valid & hour_mask[:, None]
        if not m.any():
            return {"regime": label, "hours": int(hour_mask.sum()), "mae": np.nan, "median_ae": np.nan, "mean_lmp": np.nan}
        return {
            "regime": label,
            "hours": int(hour_mask.sum()),
            "mae": float(err[m].mean()),
            "median_ae": float(np.median(err[m])),
            "mean_lmp": float(lmp_true[m].mean()),
        }

    all_hours = np.ones(len(lmp_true), dtype=bool)
    return pd.DataFrame(
        [
            stat(all_hours, "ALL"),
            stat(~congested, "uncongested"),
            stat(congested, "congested"),
            stat(curtailed, "curtailment > 0 (lambda=0)"),
            stat(~curtailed & ~reserve & ~shed, "clean: no curtail/reserve/shed"),
            stat(reserve, "reserve price > 0"),
            stat(shortfall, "reserve SHORTFALL > 0"),
            stat(~shortfall & ~shed, "MECHANISM-VALID (no shortfall/shed)"),
            stat(shed, "load shedding"),
        ]
    )


def report_diagnostics(state: dict, result: dict, curves: dict):
    """Re-measure the four structural facts the method is built on."""
    lmp_true, n_binding = state["lmp"], result["n_binding"]
    spread = np.nanmax(lmp_true, axis=1) - np.nanmin(lmp_true, axis=1)
    lam_true = np.nanmean(lmp_true, axis=1)
    uncong = n_binding == 0

    print("\n--- structural diagnostics ---")
    u = uncong & np.isfinite(spread)
    print(f"1. no binding branch -> LMP spatially uniform : {100 * (spread[u] < 0.01).mean():5.1f}%  ({u.sum()} hours)")

    z = np.isfinite(lam_true) & (np.abs(lam_true) < 1e-6)
    print(f"2. lambda == 0 hours with curtailment > 0     : {100 * (state['curtailment'][z] > 0).mean():5.1f}%  ({z.sum()} hours)")

    uids, dispatch, up = state["uids"], state["dispatch"], state["up_capable"]
    hits, ranks, unmatched = [], [], []
    for t in np.where(uncong & (lam_true > 1e-6) & np.isfinite(lam_true))[0]:
        cands = set()
        for i in np.where(up[t])[0]:
            cands.update(candidate_lambdas(curves[uids[i]], dispatch[t, i]))
        if not cands:
            continue
        c = np.array(sorted(cands))
        j = int(np.argmin(np.abs(c - lam_true[t])))
        hit = abs(c[j] - lam_true[t]) < PRICE_TOL
        hits.append(hit)
        if hit:
            ranks.append(j / max(len(c) - 1, 1))
        else:
            unmatched.append(t)
    hits = np.array(hits)
    print(f"3. lambda equals a candidate segment cost     : {100 * hits.mean():5.1f}%  ({len(hits)} hours)")
    print(f"   median rank of the matching candidate      : {np.median(ranks):5.3f}  (0=cheapest, 1=dearest)")

    if unmatched:
        um = np.array(unmatched)
        rp = state["reserve_price"]
        matched_hours = np.array([t for t, h in zip(np.where(uncong & (lam_true > 1e-6) & np.isfinite(lam_true))[0], hits) if h])
        print(f"4. unmatched hours with reserve price > 0     : {100 * (rp[um] > 1e-6).mean():5.1f}%  (mean ${rp[um].mean():.2f})")
        print(f"   matched   hours with reserve price > 0     : {100 * (rp[matched_hours] > 1e-6).mean():5.1f}%  (mean ${rp[matched_hours].mean():.2f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Recover LMPs from the DC-OPF dual and compare against Prescient.")
    parser.add_argument("--pcm-run", type=str, default=data_prep.DEFAULT_PCM_RUN)
    parser.add_argument("--method", choices=["ls", "oracle", "both"], default="both")
    parser.add_argument("--binding-threshold", type=float, default=BINDING_THRESHOLD)
    parser.add_argument("--ridge", type=float, default=RIDGE,
                        help="Ridge weight on congestion prices; guards the near-singular dual solve.")
    parser.add_argument("--price-threshold", type=float, default=PRICE_THRESHOLD,
                        help="Prescient's `price_threshold` (energy-price cap, $/MWh) from the PCM run config. "
                             "Pass 0 to disable the cap and the load-shedding special case.")
    parser.add_argument("--nu-max", type=float, default=None,
                        help="Optional cap on a branch's congestion price. Default: uncapped, matching "
                             "transmission_price_threshold=None in the PCM config. Inert on this run.")
    parser.add_argument("--report-diagnostics", action="store_true", help="Re-measure the structural facts the method relies on.")
    parser.add_argument("--csv-out", type=str, default=None, help="Write the per-bus MAE table to this path.")
    parser.add_argument("--top-n", type=int, default=73, help="How many buses to print, worst MAE first.")
    args = parser.parse_args()

    if not _HAVE_SCIPY:
        print("NOTE: scipy is unavailable; falling back to clipped unconstrained least squares. "
              "The sign constraints on nu are then only approximately enforced.\n")

    bus_ids, bus_index, _, _ = data_prep.load_topology()
    b_bus, b_f, line_ids, line_ratings = data_prep.build_susceptance_matrices(bus_index)
    ref_node, ref_bus_id = data_prep.reference_node(bus_index=bus_index)
    ptdf = data_prep.build_ptdf(b_bus, b_f, ref_node)

    curves = thermal_cost_curves()
    run_dir = data_prep.resolve_pcm_runs(args.pcm_run)[0]
    state = load_pcm_state(run_dir, curves, bus_ids, line_ids, line_ratings)

    print(f"PCM run        : {run_dir.name}")
    print(f"hours          : {len(state['keys'])}")
    print(f"buses          : {len(bus_ids)}   branches: {len(line_ids)}   ref bus: {ref_bus_id}")
    print(f"thermal units  : {len(curves)}   (segments each: {sorted({len(c['segments']) for c in curves.values()})})")
    print(f"observed LMP   : mean ${np.nanmean(state['lmp']):.2f}  min ${np.nanmin(state['lmp']):.2f}  "
          f"max ${np.nanmax(state['lmp']):.2f}  exactly zero {100 * (state['lmp'] == 0).mean():.1f}%")

    methods = ["ls", "oracle"] if args.method == "both" else [args.method]
    results, tables = {}, {}

    for method in methods:
        res = derive_lmp(state, ptdf, curves, bus_index, method=method,
                         binding_threshold=args.binding_threshold, ridge=args.ridge,
                         price_threshold=(args.price_threshold if args.price_threshold else None),
                         nu_max=args.nu_max)
        results[method] = res
        tables[method] = per_bus_mae(res["lmp_hat"], state["lmp"], bus_ids)

        label = "least squares (no learning)" if method == "ls" else "oracle lambda (classifier ceiling)"
        print(f"\n{'=' * 74}\nMETHOD: {method}  --  {label}\n{'=' * 74}")
        print(stratified_summary(res["lmp_hat"], state, res).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    if args.report_diagnostics:
        report_diagnostics(state, results[methods[0]], curves)

    main_method = "oracle" if "oracle" in tables else methods[0]
    table = tables[main_method].copy()
    if len(tables) == 2:
        table = table.merge(
            tables["ls"][["bus_id", "mae"]].rename(columns={"mae": "mae_ls"}), on="bus_id"
        ).rename(columns={"mae": "mae_oracle"})
        table = table[["bus_id", "n_hours", "mean_lmp", "mae_ls", "mae_oracle", "median_ae", "bias", "max_abs_err"]]

    sort_col = "mae_oracle" if "mae_oracle" in table else "mae"
    table = table.sort_values(sort_col, ascending=False)

    print(f"\n{'=' * 74}\nPER-BUS MAE ($/MWh), worst first\n{'=' * 74}")
    print(table.head(args.top_n).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print(f"\nacross all {len(table)} buses, {sort_col}:")
    print(f"  mean  ${table[sort_col].mean():7.3f}   median ${table[sort_col].median():7.3f}")
    print(f"  best  ${table[sort_col].min():7.3f} (bus {int(table.loc[table[sort_col].idxmin(), 'bus_id'])})"
          f"   worst ${table[sort_col].max():7.3f} (bus {int(table.loc[table[sort_col].idxmax(), 'bus_id'])})")
    print(f"  spread across buses is {'large' if table[sort_col].max() > 3 * table[sort_col].min() else 'small'} "
          "-- a wide spread means the congestion term, not lambda, dominates the error")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nwrote per-bus MAE table to {out}")


if __name__ == "__main__":
    main()
