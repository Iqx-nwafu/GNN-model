# transient_switch_tsnet_v2.py
#
# Batch transient simulation of irrigation-group switching (GroupK -> GroupK+1) using TSNet,
# driven by time-varying node withdrawals derived from Groups.xlsx.
#
# Output improvement (per user request)
#   - For each switch, automatically export ALL nodes that belong to BOTH the before-group
#     and after-group flow paths (the full node list in column A of both sheets), so that
#     when switching 3->4, 6->7, 9->10, ... the new branch nodes (e.g., J41.., J71..)
#     are visible in CSV.
#   - Optional: export a global, consistent column-set across all switches.
#
# Dependencies
#   pip install tsnet wntr pandas numpy networkx
#
# Inputs (user-provided)
#   INP:    E:\test\pythonProject\gnn\temp.inp
#   Groups: C:\Users\Administrator\Desktop\Gnn\Groups.xlsx

import math
import re
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import wntr
import tsnet
import networkx as nx

# -----------------------------
# 0) Paths (edit if needed)
# -----------------------------
INP_FILE = r"E:\test\pythonProject\gnn\temp.inp"
GROUPS_XLSX = r"C:\Users\Administrator\Desktop\Gnn\Groups.xlsx"


# -----------------------------
# 1) Utilities
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


# -----------------------------
# 2) Groups.xlsx parsing
# -----------------------------

def parse_group_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized columns: node, q_lps, state."""
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

    # Keep only Jxx node rows
    out = out[out["node"].str.match(r"^J\d+$", na=False)].copy()
    return out


def load_groups(groups_xlsx: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(groups_xlsx)
    groups = {}
    for sh in xls.sheet_names:
        df = pd.read_excel(groups_xlsx, sheet_name=sh)
        groups[sh] = parse_group_sheet(df)
    return groups


def demands_from_group_df(group_df: pd.DataFrame) -> dict[str, float]:
    """Compute node withdrawals (m3/s) from group sheet.

    Semantics (as discussed):
      - Trunk node Jk (k<10): state open/close indicates if branch inlet has water.
      - Branch nodes Jk1,Jk2,... (>=10):
          state=open => withdrawal occurs
          state=close => pass-through only
      - Column B is throughflow at node (L/s), ordered by node index within each branch.
      - withdrawal at node i: max(q_i - q_{i+1}, 0), only if state=open.
    """
    info = {r["node"]: (float(r["q_lps"]), r["state"]) for _, r in group_df.iterrows()}

    branch_nodes = [n for n in info.keys() if (_node_num(n) is not None and _node_num(n) >= 10)]

    branches: dict[int, list[str]] = {}
    for n in branch_nodes:
        num = _node_num(n)
        trunk_id = num // 10
        branches.setdefault(trunk_id, []).append(n)

    dem: dict[str, float] = {}

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
                dem[n] = take_lps / 1000.0  # m3/s

    return dem


# -----------------------------
# 3) WNTR helpers: set base demands
# -----------------------------

def wntr_set_all_demands_zero(wn):
    for n in wn.junction_name_list:
        j = wn.get_node(n)
        if len(j.demand_timeseries_list) == 0:
            j.add_demand(0.0)
        j.demand_timeseries_list[0].base_value = 0.0


def wntr_apply_demands(wn, demand_m3s: dict[str, float]):
    wntr_set_all_demands_zero(wn)
    for n, q in demand_m3s.items():
        if n in wn.junction_name_list:
            j = wn.get_node(n)
            if len(j.demand_timeseries_list) == 0:
                j.add_demand(q)
            else:
                j.demand_timeseries_list[0].base_value = q


# -----------------------------
# 4) Linear ramp of PA(t) (pulse_coeff)
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


# -----------------------------
# 5) Map node -> upstream link via shortest path from source
# -----------------------------

def build_upstream_link_map(
    wn: wntr.network.WaterNetworkModel,
    source_node: str,
    target_nodes: list[str],
) -> dict[str, str]:
    """Return mapping node -> upstream link_id (last edge on shortest path from source)."""
    G = nx.Graph()
    for lid, link in wn.links():
        u = link.start_node_name
        v = link.end_node_name
        w = getattr(link, "length", 1.0)
        G.add_edge(u, v, link_id=lid, weight=w)

    mp: dict[str, str] = {}

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
# 6) Watch-node selection (OUTPUT FIX)
# -----------------------------

def _ensure_trunk_nodes(nodes: set[str]) -> set[str]:
    """If branch nodes Jk1... exist, ensure corresponding trunk node Jk is also included."""
    extra = set()
    for n in nodes:
        num = _node_num(n)
        if num is None:
            continue
        if num >= 10:
            extra.add(f"J{num // 10}")
    return nodes | extra


def watch_nodes_from_two_groups(
    g_before: pd.DataFrame,
    g_after: pd.DataFrame,
    source_node: str,
    extra: Optional[Iterable[str]] = None,
) -> list[str]:
    """Union of *all* nodes listed in col-A of both sheets, plus source + optional extra."""
    s = set(map(str, g_before["node"].tolist())) | set(map(str, g_after["node"].tolist()))
    s.add(source_node)
    if extra:
        s |= set(map(str, extra))
    s = _ensure_trunk_nodes(s)
    return sorted(s, key=_node_sort_key)


def watch_nodes_global(groups_cache: dict[str, pd.DataFrame], source_node: str) -> list[str]:
    """Union of all nodes across all group sheets (consistent columns for every switch)."""
    s = {source_node}
    for df in groups_cache.values():
        if "node" in df.columns:
            s |= set(map(str, df["node"].tolist()))
    s = _ensure_trunk_nodes(s)
    return sorted(s, key=_node_sort_key)


# -----------------------------
# 7) Core: simulate one switch
# -----------------------------

def simulate_group_switch(
    inp_file: str,
    groups_xlsx: str,
    before_sheet: str,
    after_sheet: str,
    source_node: str = "J0",
    t_pre: float = 5.0,
    t_switch: float = 100.0,
    t_post: float = 5.0,
    dt_user: float = 0.01,
    wavespeed: float = 1200.0,
    engine: str = "DD",
    watch_nodes: Optional[list[str]] = None,
    out_prefix: str = "groupK_to_groupK1",
    groups_cache: Optional[dict[str, pd.DataFrame]] = None,
    verbose: bool = False,
):
    groups = groups_cache if groups_cache is not None else load_groups(groups_xlsx)
    if before_sheet not in groups or after_sheet not in groups:
        raise KeyError(f"Sheet missing: {before_sheet} or {after_sheet}. Available: {list(groups.keys())}")

    g0 = groups[before_sheet]
    g1 = groups[after_sheet]

    d0 = demands_from_group_df(g0)
    d1 = demands_from_group_df(g1)

    change_nodes = sorted(set(d0.keys()) | set(d1.keys()), key=_node_sort_key)

    if verbose:
        print("\n--- Demand summary (m3/s) ---")
        print("Before:", d0)
        print("After :", d1)
        print("Total before:", sum(d0.values()), "Total after:", sum(d1.values()))

    # (A) Steady state for BEFORE
    wn0 = wntr.network.WaterNetworkModel(inp_file)
    wntr_apply_demands(wn0, d0)
    res0 = wntr.sim.EpanetSimulator(wn0).run_sim()
    p0 = res0.node["pressure"].iloc[0]

    # (B) TSNet model
    tm = tsnet.network.TransientModel(inp_file)
    tm.set_wavespeed(wavespeed)

    tf = t_pre + t_switch + t_post

    from tsnet.network.discretize import max_time_step

    dt_suggest = float(max_time_step(tm))
    dt = min(float(dt_user), dt_suggest)
    tm.set_time(tf, dt=dt)

    # Apply BEFORE base demands
    wntr_apply_demands(tm, d0)

    t = np.arange(0.0, tf + tm.time_step * 0.5, tm.time_step)

    # Preload pulse arrays
    for n in change_nodes:
        node = tm.get_node(n)
        node.pulse_status = False

        qb = float(d0.get(n, 0.0))
        qa = float(d1.get(n, 0.0))

        if qb > 0.0 and qa <= 0.0:
            node.pulse_coeff = linear_ramp(t, t_start=t_pre, duration=t_switch, y0=0.0, y1=-1.0)
            node.pulse_status = True
        elif qb <= 0.0 and qa > 0.0:
            node.pulse_coeff = linear_ramp(t, t_start=t_pre, duration=t_switch, y0=-1.0, y1=0.0)
            node.pulse_status = True
        elif qb > 0.0 and qa > 0.0:
            r = qa / qb
            node.pulse_coeff = linear_ramp(t, t_start=t_pre, duration=t_switch, y0=0.0, y1=(r - 1.0))
            node.pulse_status = True

    tm = tsnet.simulation.Initializer(tm, 0.0, engine)

    # Opening nodes: set demand_coeff such that q_after = k0*sqrt(p_before)
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

    # (C) OUTPUT FIX: watch nodes auto = all nodes in both group sheets
    if watch_nodes is None:
        watch_nodes = watch_nodes_from_two_groups(g0, g1, source_node=source_node, extra=change_nodes)

    # Filter watch_nodes to those that exist in the model
    existing = set(getattr(tm, "node_name_list", []))
    watch_nodes = [n for n in watch_nodes if n in existing]

    # Export head + discharge
    head_df = pd.DataFrame(index=t)
    discharge_df = pd.DataFrame(index=t)

    for n in watch_nodes:
        node = tm.get_node(n)

        head = np.array(getattr(node, "_head"))
        if len(head) < len(t):
            head = np.pad(head, (0, len(t) - len(head)), constant_values=np.nan)
        head_df[n] = head[: len(t)]

        dq = np.array(getattr(node, "demand_discharge", np.full(len(t), np.nan)))
        if len(dq) < len(t):
            dq = np.pad(dq, (0, len(t) - len(dq)), constant_values=np.nan)
        discharge_df[n] = dq[: len(t)]

    head_df.index.name = "t_s"
    discharge_df.index.name = "t_s"

    out_head = Path(f"{out_prefix}_head.csv")
    out_dis = Path(f"{out_prefix}_discharge.csv")
    out_head.parent.mkdir(parents=True, exist_ok=True)
    head_df.to_csv(out_head, encoding="utf-8-sig")
    discharge_df.to_csv(out_dis, encoding="utf-8-sig")

    # Export throughflow
    upstream_map = build_upstream_link_map(wn0, source_node=source_node, target_nodes=watch_nodes)

    through_df = pd.DataFrame(index=t)
    for n in watch_nodes:
        lid = upstream_map.get(n)
        if not lid:
            continue
        try:
            link = tm.get_link(lid)
        except Exception:
            continue

        if getattr(link, "end_node_name", None) == n:
            q = np.array(getattr(link, "end_node_flowrate", np.full(len(t), np.nan)))
        elif getattr(link, "start_node_name", None) == n:
            q = np.array(getattr(link, "start_node_flowrate", np.full(len(t), np.nan)))
        else:
            continue

        if len(q) < len(t):
            q = np.pad(q, (0, len(t) - len(q)), constant_values=np.nan)
        through_df[n] = q[: len(t)]

    through_df.index.name = "t_s"
    out_through = Path(f"{out_prefix}_throughflow.csv")
    through_df.to_csv(out_through, encoding="utf-8-sig")

    return tm, head_df, discharge_df, through_df, d0, d1, watch_nodes


# -----------------------------
# 8) Batch runner: traverse all 29 switches
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


def ordered_group_sheets(groups_xlsx: str) -> list[tuple[int, str]]:
    sheet_names = pd.ExcelFile(groups_xlsx).sheet_names
    pairs: list[tuple[int, str]] = []
    for sh in sheet_names:
        gid = extract_group_id(sh)
        if gid is None:
            continue
        pairs.append((gid, sh))

    seen = set()
    uniq: list[tuple[int, str]] = []
    for gid, sh in sorted(pairs, key=lambda x: x[0]):
        if gid in seen:
            continue
        seen.add(gid)
        uniq.append((gid, sh))
    return uniq


def simulate_all_29_switches(
    inp_file: str,
    groups_xlsx: str,
    source_node: str = "J0",
    t_pre: float = 5.0,
    t_switch: float = 100.0,
    t_post: float = 5.0,
    dt_user: float = 0.01,
    wavespeed: float = 1200.0,
    engine: str = "DD",
    output_dir: str = "switch_outputs",
    watch_scope: str = "pair",  # 'pair' or 'global'
    include_cycle_last_to_first: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups_cache = load_groups(groups_xlsx)

    ordered = ordered_group_sheets(groups_xlsx)
    if len(ordered) < 2:
        raise ValueError(f"Need at least 2 group sheets. Found {len(ordered)}")

    switches: list[tuple[int, str, int, str]] = []
    for (gid0, sh0), (gid1, sh1) in zip(ordered[:-1], ordered[1:]):
        switches.append((gid0, sh0, gid1, sh1))
    if include_cycle_last_to_first:
        gid0, sh0 = ordered[-1]
        gid1, sh1 = ordered[0]
        switches.append((gid0, sh0, gid1, sh1))

    global_watch = None
    if str(watch_scope).lower() == "global":
        global_watch = watch_nodes_global(groups_cache, source_node=source_node)

    summary_rows = []

    for k, (gid0, sh0, gid1, sh1) in enumerate(switches, start=1):
        prefix = out_dir / f"G{gid0:02d}_to_G{gid1:02d}"
        print(f"===== Switch {k}/{len(switches)}: {sh0} -> {sh1} =====")

        try:
            tm, head_df, discharge_df, through_df, d0, d1, wn_watch = simulate_group_switch(
                inp_file=inp_file,
                groups_xlsx=groups_xlsx,
                before_sheet=sh0,
                after_sheet=sh1,
                source_node=source_node,
                t_pre=t_pre,
                t_switch=t_switch,
                t_post=t_post,
                dt_user=dt_user,
                wavespeed=wavespeed,
                engine=engine,
                watch_nodes=global_watch,  # None -> auto per pair
                out_prefix=str(prefix),
                groups_cache=groups_cache,
                verbose=False,
            )

            min_head = float(np.nanmin(head_df.to_numpy())) if head_df.size else np.nan
            max_head = float(np.nanmax(head_df.to_numpy())) if head_df.size else np.nan
            nan_frac = float(np.isnan(head_df.to_numpy()).mean()) if head_df.size else np.nan

            summary_rows.append(
                {
                    "switch_idx": k,
                    "from_gid": gid0,
                    "from_sheet": sh0,
                    "to_gid": gid1,
                    "to_sheet": sh1,
                    "total_before_m3s": float(sum(d0.values())),
                    "total_after_m3s": float(sum(d1.values())),
                    "t_pre_s": float(t_pre),
                    "t_switch_s": float(t_switch),
                    "t_post_s": float(t_post),
                    "dt_s": float(getattr(tm, "time_step", np.nan)),
                    "min_head_m": min_head,
                    "max_head_m": max_head,
                    "head_nan_frac": nan_frac,
                    "n_watch_nodes": int(len(wn_watch)),
                    "watch_scope": str(watch_scope),
                    "out_prefix": str(prefix),
                    "status": "ok",
                    "error": "",
                }
            )

        except Exception as e:
            summary_rows.append(
                {
                    "switch_idx": k,
                    "from_gid": gid0,
                    "from_sheet": sh0,
                    "to_gid": gid1,
                    "to_sheet": sh1,
                    "total_before_m3s": np.nan,
                    "total_after_m3s": np.nan,
                    "t_pre_s": float(t_pre),
                    "t_switch_s": float(t_switch),
                    "t_post_s": float(t_post),
                    "dt_s": np.nan,
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
            print(f"[ERROR] Switch {k} failed: {e!r}")
            continue

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "switch_summary.csv"
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

    # NOTE:
    #   watch_scope='pair'  -> 每次切换自动输出“前一组sheet节点 + 后一组sheet节点”，满足你提出的 3->4、6->7、... 可见新支管。
    #   watch_scope='global'-> 所有切换输出同一套列（所有组sheet节点并集），便于后续拼接/建模，但文件更宽。

    simulate_all_29_switches(
        inp_file=INP_FILE,
        groups_xlsx=GROUPS_XLSX,
        source_node="J0",
        t_pre=5.0,
        t_switch=100.0,
        t_post=5.0,
        dt_user=0.01,
        wavespeed=1200.0,
        engine="DD",
        output_dir="switch_outputs",
        watch_scope="pair",
        include_cycle_last_to_first=False,
    )
