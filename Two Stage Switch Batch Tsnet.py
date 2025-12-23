# two_stage_switch_batch_tsnet.py
#
# Batch transient simulation for irrigation-group switching using TSNet (MOC), with:
#   - Two-stage linear CLOSING (fast then slow) for closing nodes
#   - Linear OPENING for opening nodes
#   - Total switch duration fixed at 60 s
#   - Sweep fast-closing time: 1..20 s (20 rates) for EACH switch
#   - Time step target: 0.1 s (internal dt respects TSNet stability; output is resampled to 0.1 s)
#   - Output columns include ALL nodes listed in column A of BOTH the before/after group sheets
#
# Authoring note:
#   This script assumes the semantics we previously used:
#   - Groups.xlsx sheet has: Column A node list (Jxx), Column B throughflow at node (L/s), Column C open/close
#   - Node withdrawals are derived from throughflow differences within each branch, only at nodes with state=open
#
# Dependencies
#   pip install tsnet wntr pandas numpy networkx

import math
import re
from pathlib import Path
from typing import Optional, Iterable, Dict, List, Tuple

import numpy as np
import pandas as pd
import wntr
import tsnet
import networkx as nx

# -----------------------------
# 0) User paths
# -----------------------------
INP_FILE = r"E:\test\pythonProject\gnn\temp.inp"
GROUPS_XLSX = r"C:\Users\Administrator\Desktop\Gnn\Groups.xlsx"

# -----------------------------
# 1) Basic helpers
# -----------------------------

def read_inp_units(inp_path: str) -> Optional[str]:
    """Parse [OPTIONS] UNITS from an EPANET INP file (best-effort)."""
    try:
        txt = Path(inp_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        try:
            txt = Path(inp_path).read_text(encoding="gbk", errors="ignore").splitlines()
        except Exception:
            return None

    in_options = False
    for line in txt:
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        if s.upper().startswith("[OPTIONS]"):
            in_options = True
            continue
        if in_options and s.startswith("["):
            break
        if in_options:
            parts = re.split(r"\s+", s)
            if len(parts) >= 2 and parts[0].upper() == "UNITS":
                return parts[1].upper()
    return None


def _norm_state(x) -> str:
    s = str(x).strip().lower()
    if s in ("open", "opened", "1", "true", "on", "开启", "开"):
        return "open"
    if s in ("close", "closed", "0", "false", "off", "关闭", "关"):
        return "close"
    return "close"


def _node_num(n: str) -> Optional[int]:
    m = re.match(r"^J(\d+)$", str(n).strip())
    return int(m.group(1)) if m else None


def _node_sort_key(n: str):
    num = _node_num(n)
    return (0, num) if num is not None else (1, str(n))


def _ensure_trunk_nodes(nodes: set[str]) -> set[str]:
    """If branch nodes Jk1... exist, ensure corresponding trunk node Jk is included."""
    extra = set()
    for n in nodes:
        num = _node_num(n)
        if num is None:
            continue
        if num >= 10:
            extra.add(f"J{num // 10}")
    return nodes | extra


# -----------------------------
# 2) Groups.xlsx parsing
# -----------------------------

def parse_group_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to columns: node, q_lps, state."""
    if df.shape[1] < 3:
        raise ValueError("Each group sheet needs at least 3 columns: node / flow(L/s) / open-close")

    cols = list(df.columns)

    def pick_col(keywords, fallback_idx):
        for c in cols:
            cs = str(c)
            if any(k in cs for k in keywords):
                return c
        return cols[fallback_idx]

    c_node = pick_col(["水流经过节点", "节点", "Node"], 0)
    c_q = pick_col(["节点处流量", "流量", "L/s", "l/s", "Flow"], 1)
    c_st = pick_col(["启闭情况", "启闭", "状态", "State"], 2)

    out = df[[c_node, c_q, c_st]].copy()
    out.columns = ["node", "q_lps", "state"]

    out["node"] = out["node"].astype(str).str.strip()
    out["q_lps"] = pd.to_numeric(out["q_lps"], errors="coerce").fillna(0.0)
    out["state"] = out["state"].apply(_norm_state)

    out = out[out["node"].str.match(r"^J\d+$", na=False)].copy()
    return out


def load_groups(groups_xlsx: str) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(groups_xlsx)
    groups: Dict[str, pd.DataFrame] = {}
    for sh in xls.sheet_names:
        df = pd.read_excel(groups_xlsx, sheet_name=sh)
        groups[sh] = parse_group_sheet(df)
    return groups


def demands_from_group_df(group_df: pd.DataFrame) -> Dict[str, float]:
    """Compute node withdrawals (m3/s) from a group sheet."""
    info = {r["node"]: (float(r["q_lps"]), r["state"]) for _, r in group_df.iterrows()}

    branch_nodes = [n for n in info.keys() if (_node_num(n) is not None and _node_num(n) >= 10)]

    branches: Dict[int, List[str]] = {}
    for n in branch_nodes:
        num = _node_num(n)
        trunk_id = num // 10
        branches.setdefault(trunk_id, []).append(n)

    dem: Dict[str, float] = {}

    for trunk_id, nodes in branches.items():
        trunk = f"J{trunk_id}"
        trunk_state = info.get(trunk, (0.0, "open"))[1]
        if trunk_state == "close":
            continue

        nodes_sorted = sorted(nodes, key=_node_sort_key)
        q = [info[n][0] for n in nodes_sorted]  # L/s
        st = [info[n][1] for n in nodes_sorted]

        for i, n in enumerate(nodes_sorted):
            qi = q[i]
            qn = q[i + 1] if i < len(nodes_sorted) - 1 else 0.0
            take_lps = max(qi - qn, 0.0)
            if st[i] == "open" and take_lps > 0:
                dem[n] = take_lps / 1000.0

    return dem


def watch_nodes_from_two_groups(
    g_before: pd.DataFrame,
    g_after: pd.DataFrame,
    source_node: str,
    extra: Optional[Iterable[str]] = None,
) -> List[str]:
    """Union of all nodes listed in column A of both sheets, plus source and optional extra."""
    s = set(map(str, g_before["node"].tolist())) | set(map(str, g_after["node"].tolist()))
    s.add(source_node)
    if extra:
        s |= set(map(str, extra))
    s = _ensure_trunk_nodes(s)
    return sorted(s, key=_node_sort_key)


def watch_nodes_global(groups_cache: Dict[str, pd.DataFrame], source_node: str) -> List[str]:
    """Union of all nodes across all group sheets (consistent columns for every switch)."""
    s = {source_node}
    for df in groups_cache.values():
        s |= set(map(str, df["node"].tolist()))
    s = _ensure_trunk_nodes(s)
    return sorted(s, key=_node_sort_key)


# -----------------------------
# 3) WNTR helpers
# -----------------------------

def wntr_set_all_demands_zero(wn):
    for n in wn.junction_name_list:
        j = wn.get_node(n)
        if len(j.demand_timeseries_list) == 0:
            j.add_demand(0.0)
        j.demand_timeseries_list[0].base_value = 0.0


def wntr_apply_demands(wn, demand_m3s: Dict[str, float]):
    wntr_set_all_demands_zero(wn)
    for n, q in demand_m3s.items():
        if n in wn.junction_name_list:
            j = wn.get_node(n)
            if len(j.demand_timeseries_list) == 0:
                j.add_demand(q)
            else:
                j.demand_timeseries_list[0].base_value = q


# -----------------------------
# 4) PA(t) constructions
# -----------------------------

def linear_ramp(t: np.ndarray, t_start: float, duration: float, y0: float, y1: float) -> np.ndarray:
    """Piecewise linear: y=y0 before, ramps during, y=y1 after."""
    y = np.full_like(t, fill_value=y0, dtype=float)
    t_end = t_start + duration
    mid = (t >= t_start) & (t <= t_end)
    if duration <= 0:
        y[t >= t_start] = y1
        return y
    y[mid] = y0 + (y1 - y0) * (t[mid] - t_start) / duration
    y[t > t_end] = y1
    return y


def two_stage_close_PA(
    t: np.ndarray,
    t_start: float,
    T: float,
    t_fast: float,
    gamma: float = 0.9,
) -> np.ndarray:
    """Two-stage closing PA(t): 0 -> -gamma in t_fast, then -gamma -> -1 in (T-t_fast)."""
    t1 = t_start + float(t_fast)
    t2 = t_start + float(T)

    PA = np.zeros_like(t, dtype=float)

    PA[t < t_start] = 0.0

    # stage 1
    m1 = (t >= t_start) & (t <= t1)
    if t_fast > 0:
        PA[m1] = (0.0) + (-gamma - 0.0) * (t[m1] - t_start) / float(t_fast)
    else:
        PA[m1] = -gamma

    # stage 2
    m2 = (t > t1) & (t <= t2)
    if (T - t_fast) > 0:
        PA[m2] = (-gamma) + (-1.0 + gamma) * (t[m2] - t1) / float(T - t_fast)
    else:
        PA[m2] = -1.0

    PA[t > t2] = -1.0
    return PA


# -----------------------------
# 5) Node -> upstream link mapping (for throughflow export)
# -----------------------------

def build_upstream_link_map(
    wn: wntr.network.WaterNetworkModel,
    source_node: str,
    target_nodes: List[str],
) -> Dict[str, str]:
    """Return mapping node -> upstream link_id (last edge on shortest path from source)."""
    G = nx.Graph()
    for lid, link in wn.links():
        u = link.start_node_name
        v = link.end_node_name
        w = getattr(link, "length", 1.0)
        G.add_edge(u, v, link_id=lid, weight=w)

    mp: Dict[str, str] = {}
    for n in target_nodes:
        if n == source_node or n not in G or source_node not in G:
            continue
        try:
            path = nx.shortest_path(G, source_node, n, weight="weight")
        except Exception:
            continue
        if len(path) < 2:
            continue
        a, b = path[-2], path[-1]
        mp[n] = G[a][b]["link_id"]
    return mp


# -----------------------------
# 6) Resampling to a fixed output dt (0.1 s)
# -----------------------------

def _interp_series(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """1D linear interpolation with NaN safety."""
    y = np.asarray(y_src, dtype=float)
    t = np.asarray(t_src, dtype=float)
    tgt = np.asarray(t_tgt, dtype=float)

    if len(y) != len(t):
        m = min(len(y), len(t))
        y = y[:m]
        t = t[:m]

    ok = np.isfinite(y) & np.isfinite(t)
    if ok.sum() < 2:
        return np.full_like(tgt, np.nan, dtype=float)

    return np.interp(tgt, t[ok], y[ok])


def resample_dataframe(df: pd.DataFrame, t_src: np.ndarray, t_tgt: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=t_tgt)
    for c in df.columns:
        out[c] = _interp_series(t_src, df[c].to_numpy(), t_tgt)
    out.index.name = df.index.name
    return out


# -----------------------------
# 7) Simulate one switch with one fast-closing time
# -----------------------------

def simulate_group_switch_two_stage(
    inp_file: str,
    groups_cache: Dict[str, pd.DataFrame],
    before_sheet: str,
    after_sheet: str,
    fast_time_s: int,
    gamma: float = 0.9,
    source_node: str = "J0",
    t_pre: float = 5.0,
    t_switch: float = 60.0,
    t_post: float = 5.0,
    dt_output: float = 0.1,
    wavespeed: float = 1200.0,
    engine: str = "DD",
    watch_scope: str = "pair",  # 'pair' or 'global'
    global_watch: Optional[List[str]] = None,
    out_prefix: str = "G01_to_G02/fast_01s",
    verbose: bool = False,
):
    if before_sheet not in groups_cache or after_sheet not in groups_cache:
        raise KeyError(f"Sheet missing: {before_sheet} or {after_sheet}")

    g0 = groups_cache[before_sheet]
    g1 = groups_cache[after_sheet]

    d0 = demands_from_group_df(g0)
    d1 = demands_from_group_df(g1)

    change_nodes = sorted(set(d0.keys()) | set(d1.keys()), key=_node_sort_key)

    # --- Steady state for BEFORE (pressure for opening-node demand_coeff) ---
    wn0 = wntr.network.WaterNetworkModel(inp_file)
    wntr_apply_demands(wn0, d0)
    res0 = wntr.sim.EpanetSimulator(wn0).run_sim()
    p0 = res0.node["pressure"].iloc[0]

    # --- TSNet transient model ---
    tm = tsnet.network.TransientModel(inp_file)
    tm.set_wavespeed(wavespeed)

    tf = float(t_pre + t_switch + t_post)

    # Stability dt from TSNet
    from tsnet.network.discretize import max_time_step

    dt_suggest = float(max_time_step(tm))
    dt_internal = min(float(dt_output), dt_suggest)

    tm.set_time(tf, dt=dt_internal)

    # Apply BEFORE base demands
    wntr_apply_demands(tm, d0)

    t_internal = np.arange(0.0, tf + tm.time_step * 0.5, tm.time_step)

    # Build pulse_coeff for changing nodes
    for n in change_nodes:
        node = tm.get_node(n)
        node.pulse_status = False

        qb = float(d0.get(n, 0.0))
        qa = float(d1.get(n, 0.0))

        # closing: two-stage
        if qb > 0.0 and qa <= 0.0:
            node.pulse_coeff = two_stage_close_PA(
                t_internal, t_start=t_pre, T=t_switch, t_fast=float(fast_time_s), gamma=float(gamma)
            )
            node.pulse_status = True

        # opening: linear (-1 -> 0)
        elif qb <= 0.0 and qa > 0.0:
            node.pulse_coeff = linear_ramp(t_internal, t_start=t_pre, duration=t_switch, y0=-1.0, y1=0.0)
            node.pulse_status = True

        # nonzero -> nonzero: linear scaling (0 -> r-1)
        elif qb > 0.0 and qa > 0.0:
            r = qa / qb
            node.pulse_coeff = linear_ramp(t_internal, t_start=t_pre, duration=t_switch, y0=0.0, y1=(r - 1.0))
            node.pulse_status = True

    tm = tsnet.simulation.Initializer(tm, 0.0, engine)

    # Opening nodes need demand_coeff k0 so that q_after = k0*sqrt(p_before)
    eps_p = 1e-3
    for n in change_nodes:
        qb = float(d0.get(n, 0.0))
        qa = float(d1.get(n, 0.0))
        if qb <= 0.0 and qa > 0.0:
            pb = float(p0.get(n, 0.0))
            pb = max(pb, eps_p)
            k0 = qa / math.sqrt(pb)
            setattr(tm.get_node(n), "demand_coeff", k0)

    tm = tsnet.simulation.MOCSimulator(tm, "moc_results")

    # --- Watch nodes for OUTPUT ---
    if watch_scope.lower() == "global":
        watch_nodes = global_watch if global_watch is not None else watch_nodes_global(groups_cache, source_node)
    else:
        watch_nodes = watch_nodes_from_two_groups(g0, g1, source_node=source_node, extra=change_nodes)

    # keep only nodes that exist in model
    existing = set(getattr(tm, "node_name_list", []))
    watch_nodes = [n for n in watch_nodes if n in existing]

    # --- Export (internal time) ---
    head_df = pd.DataFrame(index=t_internal)
    discharge_df = pd.DataFrame(index=t_internal)

    for n in watch_nodes:
        node = tm.get_node(n)

        head = np.array(getattr(node, "_head"))
        if len(head) < len(t_internal):
            head = np.pad(head, (0, len(t_internal) - len(head)), constant_values=np.nan)
        head_df[n] = head[: len(t_internal)]

        dq = np.array(getattr(node, "demand_discharge", np.full(len(t_internal), np.nan)))
        if len(dq) < len(t_internal):
            dq = np.pad(dq, (0, len(t_internal) - len(dq)), constant_values=np.nan)
        discharge_df[n] = dq[: len(t_internal)]

    head_df.index.name = "t_s"
    discharge_df.index.name = "t_s"

    # throughflow using upstream link flowrate at node end
    upstream_map = build_upstream_link_map(wn0, source_node=source_node, target_nodes=watch_nodes)

    through_df = pd.DataFrame(index=t_internal)
    for n in watch_nodes:
        lid = upstream_map.get(n)
        if not lid:
            continue
        try:
            link = tm.get_link(lid)
        except Exception:
            continue

        if getattr(link, "end_node_name", None) == n:
            q = np.array(getattr(link, "end_node_flowrate", np.full(len(t_internal), np.nan)))
        elif getattr(link, "start_node_name", None) == n:
            q = np.array(getattr(link, "start_node_flowrate", np.full(len(t_internal), np.nan)))
        else:
            continue

        if len(q) < len(t_internal):
            q = np.pad(q, (0, len(t_internal) - len(q)), constant_values=np.nan)
        through_df[n] = q[: len(t_internal)]

    through_df.index.name = "t_s"

    # --- Resample to fixed output dt (0.1 s) if needed ---
    t_out = np.arange(0.0, tf + 1e-12, float(dt_output))
    if abs(dt_internal - dt_output) > 1e-12:
        head_out = resample_dataframe(head_df, t_internal, t_out)
        dis_out = resample_dataframe(discharge_df, t_internal, t_out)
        thr_out = resample_dataframe(through_df, t_internal, t_out)
    else:
        head_out, dis_out, thr_out = head_df, discharge_df, through_df
        head_out.index = t_out
        dis_out.index = t_out
        thr_out.index = t_out
        head_out.index.name = "t_s"
        dis_out.index.name = "t_s"
        thr_out.index.name = "t_s"

    # --- Save ---
    out_head = Path(f"{out_prefix}_head.csv")
    out_dis = Path(f"{out_prefix}_discharge.csv")
    out_thr = Path(f"{out_prefix}_throughflow.csv")
    out_head.parent.mkdir(parents=True, exist_ok=True)

    head_out.to_csv(out_head, encoding="utf-8-sig")
    dis_out.to_csv(out_dis, encoding="utf-8-sig")
    thr_out.to_csv(out_thr, encoding="utf-8-sig")

    if verbose:
        print(
            f"Saved: {out_head.name}, {out_dis.name}, {out_thr.name} | "
            f"fast={fast_time_s}s gamma={gamma} dt_int={dt_internal:.6g} dt_out={dt_output:.6g}"
        )

    return {
        "dt_internal": dt_internal,
        "dt_output": float(dt_output),
        "t_final": tf,
        "fast_time_s": int(fast_time_s),
        "gamma": float(gamma),
        "watch_nodes": watch_nodes,
        "total_before_m3s": float(sum(d0.values())),
        "total_after_m3s": float(sum(d1.values())),
        "min_head_m": float(np.nanmin(head_out.to_numpy())) if head_out.size else np.nan,
        "max_head_m": float(np.nanmax(head_out.to_numpy())) if head_out.size else np.nan,
        "head_nan_frac": float(np.isnan(head_out.to_numpy()).mean()) if head_out.size else np.nan,
    }


# -----------------------------
# 8) Batch runner: 29 switches * 20 fast times
# -----------------------------

_GROUP_PAT = re.compile(r"(?:轮灌组|灌溉组|group)\s*0*([1-9]\d*)", re.IGNORECASE)


def extract_group_id(sheet_name: str) -> Optional[int]:
    s = str(sheet_name).strip()
    m = _GROUP_PAT.search(s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", s)
    if m2:
        return int(m2.group(1))
    return None


def ordered_group_sheets(groups_xlsx: str) -> List[Tuple[int, str]]:
    sheet_names = pd.ExcelFile(groups_xlsx).sheet_names
    pairs: List[Tuple[int, str]] = []
    for sh in sheet_names:
        gid = extract_group_id(sh)
        if gid is None:
            continue
        pairs.append((gid, sh))

    seen = set()
    uniq: List[Tuple[int, str]] = []
    for gid, sh in sorted(pairs, key=lambda x: x[0]):
        if gid in seen:
            continue
        seen.add(gid)
        uniq.append((gid, sh))
    return uniq


def simulate_all_switches_all_fast_times(
    inp_file: str,
    groups_xlsx: str,
    output_dir: str = "switch_outputs_two_stage",
    source_node: str = "J0",
    t_pre: float = 5.0,
    t_switch: float = 60.0,
    t_post: float = 5.0,
    dt_output: float = 0.1,
    fast_times: Iterable[int] = range(1, 21),
    gamma: float = 0.9,
    wavespeed: float = 1200.0,
    engine: str = "DD",
    watch_scope: str = "pair",  # 'pair' recommended for compact files; 'global' for consistent columns
    include_cycle_last_to_first: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups_cache = load_groups(groups_xlsx)
    ordered = ordered_group_sheets(groups_xlsx)
    if len(ordered) < 2:
        raise ValueError(f"Need at least 2 group sheets. Found {len(ordered)}")

    switches: List[Tuple[int, str, int, str]] = []
    for (gid0, sh0), (gid1, sh1) in zip(ordered[:-1], ordered[1:]):
        switches.append((gid0, sh0, gid1, sh1))
    if include_cycle_last_to_first and len(ordered) >= 2:
        gid0, sh0 = ordered[-1]
        gid1, sh1 = ordered[0]
        switches.append((gid0, sh0, gid1, sh1))

    global_watch = None
    if watch_scope.lower() == "global":
        global_watch = watch_nodes_global(groups_cache, source_node=source_node)

    summary_rows = []

    total_runs = len(switches) * len(list(fast_times))
    run_idx = 0

    for sw_idx, (gid0, sh0, gid1, sh1) in enumerate(switches, start=1):
        sw_folder = out_dir / f"G{gid0:02d}_to_G{gid1:02d}"
        sw_folder.mkdir(parents=True, exist_ok=True)

        for ft in fast_times:
            run_idx += 1
            prefix = sw_folder / f"fast_{int(ft):02d}s"
            print(
                f"Run {run_idx}/{total_runs} | Switch {sw_idx}/{len(switches)}: "
                f"G{gid0:02d}->{gid1:02d} | fast={ft}s gamma={gamma}"
            )

            try:
                stats = simulate_group_switch_two_stage(
                    inp_file=inp_file,
                    groups_cache=groups_cache,
                    before_sheet=sh0,
                    after_sheet=sh1,
                    fast_time_s=int(ft),
                    gamma=float(gamma),
                    source_node=source_node,
                    t_pre=float(t_pre),
                    t_switch=float(t_switch),
                    t_post=float(t_post),
                    dt_output=float(dt_output),
                    wavespeed=float(wavespeed),
                    engine=str(engine),
                    watch_scope=str(watch_scope),
                    global_watch=global_watch,
                    out_prefix=str(prefix),
                    verbose=False,
                )

                summary_rows.append(
                    {
                        "switch_idx": sw_idx,
                        "from_gid": gid0,
                        "from_sheet": sh0,
                        "to_gid": gid1,
                        "to_sheet": sh1,
                        "fast_time_s": int(ft),
                        "gamma": float(gamma),
                        "t_pre_s": float(t_pre),
                        "t_switch_s": float(t_switch),
                        "t_post_s": float(t_post),
                        "dt_output_s": float(dt_output),
                        "dt_internal_s": float(stats["dt_internal"]),
                        "total_before_m3s": float(stats["total_before_m3s"]),
                        "total_after_m3s": float(stats["total_after_m3s"]),
                        "min_head_m": float(stats["min_head_m"]),
                        "max_head_m": float(stats["max_head_m"]),
                        "head_nan_frac": float(stats["head_nan_frac"]),
                        "n_watch_nodes": int(len(stats["watch_nodes"])),
                        "watch_scope": str(watch_scope),
                        "out_prefix": str(prefix),
                        "status": "ok",
                        "error": "",
                    }
                )

            except Exception as e:
                summary_rows.append(
                    {
                        "switch_idx": sw_idx,
                        "from_gid": gid0,
                        "from_sheet": sh0,
                        "to_gid": gid1,
                        "to_sheet": sh1,
                        "fast_time_s": int(ft),
                        "gamma": float(gamma),
                        "t_pre_s": float(t_pre),
                        "t_switch_s": float(t_switch),
                        "t_post_s": float(t_post),
                        "dt_output_s": float(dt_output),
                        "dt_internal_s": np.nan,
                        "total_before_m3s": np.nan,
                        "total_after_m3s": np.nan,
                        "min_head_m": np.nan,
                        "max_head_m": np.nan,
                        "head_nan_frac": np.nan,
                        "n_watch_nodes": 0,
                        "watch_scope": str(watch_scope),
                        "out_prefix": str(prefix),
                        "status": "failed",
                        "error": repr(e),
                    }
                )
                print(f"[ERROR] Switch {gid0}->{gid1} fast={ft}s failed: {e!r}")
                continue

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "switch_fasttime_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"All done. Summary saved to: {summary_path.resolve()}")

    return summary_df


# -----------------------------
# 9) Main
# -----------------------------

if __name__ == "__main__":
    units = read_inp_units(INP_FILE)
    if units is not None:
        print(f"INP [OPTIONS] UNITS = {units}")

    # User requirements:
    #   - gamma=0.8 (fast stage completes 80% closure)
    #   - output dt=0.1 s
    #   - fast time sweep: 1..20 s
    #   - open still linear, close two-stage

    simulate_all_switches_all_fast_times(
        inp_file=INP_FILE,
        groups_xlsx=GROUPS_XLSX,
        output_dir="switch_outputs_two_stage",
        source_node="J0",
        t_pre=5.0,
        t_switch=60.0,
        t_post=5.0,
        dt_output=0.1,
        fast_times=range(1, 21),
        gamma=0.9,
        wavespeed=1200.0,
        engine="DD",
        watch_scope="pair",          # 'pair' shows both before/after branch nodes each switch
        include_cycle_last_to_first=False,
    )
