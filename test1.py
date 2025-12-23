import copy
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import wntr
import networkx as nx

# ============================================================
# 1) 路径（按你电脑上的实际路径修改）
# ============================================================
NODES_XLSX  = r"C:\Users\Administrator\Desktop\Gnn\Nodes.xlsx"
PIPES_XLSX  = r"C:\Users\Administrator\Desktop\Gnn\Pipes.xlsx"
GROUPS_XLSX = r"C:\Users\Administrator\Desktop\Gnn\Groups.xlsx"

# 如果你希望直接在当前目录运行，也可以把上面三行改成：
# NODES_XLSX = "Nodes.xlsx"
# PIPES_XLSX = "Pipes.xlsx"
# GROUPS_XLSX = "Groups.xlsx"

# ============================================================
# 2) Excel 表头映射（按你的 Nodes.xlsx / Pipes.xlsx 实际列名调整）
# ============================================================
COL = {
    # Nodes.xlsx
    "node_id": "Nodel ID",  # 注意：你原表头就是这个拼写（若不同请改）
    "x": "X",
    "y": "Y",
    "z": "Z",               # elevation (m)

    # Pipes.xlsx
    "pipe_id": "Pipe ID",
    "from": "FromNode",
    "to": "ToNode",
    "length_m": "Length_m",
    "diam_m": "Diameter_m",
    "material": "Material"  # UPVC / PE
}

# Darcy–Weisbach：绝对粗糙度（常用以 mm 计）
EPS_MM = {"UPVC": 0.0015, "PE": 0.007}  # TODO: 按你选用的值修改

# “有流量”的阈值（用于防止数值噪声）
FLOW_TOL = 1e-6  # 若仍出现“假过流”，建议提到 1e-5 或 1e-4

# ============================================================
# 3) 读取 Excel
# ============================================================
nodes_df = pd.read_excel(NODES_XLSX)
pipes_df = pd.read_excel(PIPES_XLSX)
groups_dict = pd.read_excel(GROUPS_XLSX, sheet_name=None)

# ============================================================
# 4) 建网
# ============================================================
def build_wn(nodes_df: pd.DataFrame, pipes_df: pd.DataFrame) -> wntr.network.WaterNetworkModel:
    wn = wntr.network.WaterNetworkModel()

    # --- 节点 ---
    for _, r in nodes_df.iterrows():
        nid = str(r[COL["node_id"]]).strip()
        x, y = float(r[COL["x"]]), float(r[COL["y"]])
        z = float(r[COL["z"]])

        if nid == "J0":
            # 你给定：J0 水库水头 25 m（作为固定水头边界）
            wn.add_reservoir("J0", base_head=25.0, coordinates=(x, y))
        else:
            wn.add_junction(nid, elevation=z, base_demand=0.0, coordinates=(x, y))

    # --- 管段 ---
    for _, r in pipes_df.iterrows():
        pid = str(r[COL["pipe_id"]]).strip()
        n1 = str(r[COL["from"]]).strip()
        n2 = str(r[COL["to"]]).strip()
        L = float(r[COL["length_m"]])
        D = float(r[COL["diam_m"]])
        mat = str(r[COL["material"]]).strip().upper()

        if mat not in EPS_MM:
            raise KeyError(f"Material={mat} 不在 EPS_MM 映射里，请补充：{set(EPS_MM.keys())}")

        wn.add_pipe(
            pid, n1, n2,
            length=L,
            diameter=D,
            roughness=float(EPS_MM[mat]),
            minor_loss=0.0,
            initial_status="OPEN"
        )

    wn.options.hydraulic.headloss = "D-W"
    wn.options.time.duration = 0  # 仅稳态
    # 让写出的 inp 使用公制（避免你看到 temp.inp 里英制单位“像错了一样”）
    try:
        wn.options.hydraulic.inpfile_units = "CMS"
    except Exception:
        pass

    return wn


wn_base = build_wn(nodes_df, pipes_df)


# ============================================================
# 5) 需求赋值/清零（WNTR 兼容）
# ============================================================
def set_junction_demand(junc, demand_m3s: float):
    """
    兼容 WNTR：base_demand 有时是只读，因此统一改 demand_timeseries_list[0].base_value
    """
    if len(junc.demand_timeseries_list) == 0:
        junc.add_demand(float(demand_m3s))
    else:
        junc.demand_timeseries_list[0].base_value = float(demand_m3s)


# ============================================================
# 6) Groups 解析：A=节点名，B=节点处流量(L/s)，C=启闭情况(open/close)
#    - 干管节点 J1,J2,...: open=该干管处支管有来水；close=支管进口关闭
#    - 支管节点 J11,J12,...: open=取水；close=仅过水
#    - 取水量 = max(q_i - q_{i+1}, 0)，只对 state=open 的支管节点赋需求
#    - “只到最远取水点有过流”：最远 open 的支管节点之后的管段全部 CLOSED
# ============================================================
def _node_num(n: str):
    m = re.match(r"^J(\d+)$", str(n).strip())
    return int(m.group(1)) if m else None


def _norm_state(x):
    s = str(x).strip().lower()
    if s in ("open", "opened", "1", "true", "on"):
        return "open"
    if s in ("close", "closed", "0", "false", "off"):
        return "close"
    return s


def parse_group_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    支持你现在的中文列名：
      水流经过节点 | 节点处流量L/s | 启闭情况
    也支持“列顺序就是前三列”的情况。
    """
    if df.shape[1] < 3:
        raise ValueError("Group sheet 至少需要三列：节点 / 节点处流量(L/s) / 启闭情况")

    cols = list(df.columns)

    def pick_col(keywords, fallback_idx):
        for c in cols:
            cs = str(c)
            if any(k in cs for k in keywords):
                return c
        return cols[fallback_idx]

    c_node = pick_col(["节点", "Node"], 0)
    c_q = pick_col(["流量", "Flow", "L/s", "l/s"], 1)
    c_st = pick_col(["启闭", "状态", "State", "open", "close"], 2)

    out = df[[c_node, c_q, c_st]].copy()
    out.columns = ["node", "q_lps", "state"]
    out["node"] = out["node"].astype(str).str.strip()
    out["q_lps"] = pd.to_numeric(out["q_lps"], errors="coerce").fillna(0.0)
    out["state"] = out["state"].apply(_norm_state)
    out = out.dropna(subset=["node"])
    return out


def build_edge_to_pipe_id(pipes_df: pd.DataFrame):
    """
    无向边 (u,v) -> pipe_id，用于快速找到“J1-J11 这段管段”并设置 OPEN/CLOSED。
    """
    mp = {}
    for _, r in pipes_df.iterrows():
        pid = str(r[COL["pipe_id"]]).strip()
        a = str(r[COL["from"]]).strip()
        b = str(r[COL["to"]]).strip()
        mp[frozenset((a, b))] = pid
    return mp


def set_link_open(wn: wntr.network.WaterNetworkModel, link_id: str, is_open: bool):
    if not link_id:
        return
    link = wn.get_link(link_id)
    val = "OPEN" if is_open else "CLOSED"
    if hasattr(link, "initial_status"):
        link.initial_status = val
    # 某些版本也支持 link.status
    if hasattr(link, "status"):
        try:
            link.status = wntr.network.LinkStatus.Open if is_open else wntr.network.LinkStatus.Closed
        except Exception:
            try:
                link.status = val
            except Exception:
                pass


def apply_group_boundary(wn: wntr.network.WaterNetworkModel,
                         group_df: pd.DataFrame,
                         edge2pid: dict,
                         flow_eps_lps: float = 1e-6,
                         debug: bool = False):
    g = parse_group_sheet(group_df)

    # 1) 全网 junction demand 清零
    for _, j in wn.junctions():
        set_junction_demand(j, 0.0)

    # 2) node -> (q_lps, state)
    info = {r["node"]: (float(r["q_lps"]), r["state"]) for _, r in g.iterrows()}

    # 3) 找出所有支管节点（编号>=10）
    branch_nodes = [n for n in info.keys() if (_node_num(n) is not None and _node_num(n) >= 10)]

    # 4) 按 trunk_id 分组：J11..J19 => trunk=J1；J21..J29 => trunk=J2 ...
    branches = {}
    for n in branch_nodes:
        num = _node_num(n)
        trunk_id = num // 10
        branches.setdefault(trunk_id, []).append(n)

    # 5) 逐条支管处理
    for trunk_id, nodes in branches.items():
        trunk = f"J{trunk_id}"
        trunk_state = info.get(trunk, (0.0, "open"))[1]  # 若 sheet 没列 trunk，则默认 open

        nodes_sorted = sorted(nodes, key=lambda x: _node_num(x))
        b_start = nodes_sorted[0]  # 支管起点（如 J11/J21）

        # 支管进口管段：Jk -> Jk1
        inlet_pid = edge2pid.get(frozenset((trunk, b_start)))

        if trunk_state == "close":
            # 干管 close：关进口 + 关整条支管
            set_link_open(wn, inlet_pid, False)
            for a, b in zip(nodes_sorted[:-1], nodes_sorted[1:]):
                pid = edge2pid.get(frozenset((a, b)))
                set_link_open(wn, pid, False)
            continue
        else:
            set_link_open(wn, inlet_pid, True)

        # 节点处过流量（L/s）与状态
        q = [info[n][0] for n in nodes_sorted]
        st = [info[n][1] for n in nodes_sorted]

        # 5.1 赋需求：仅对 state=open 的支管节点赋值
        for i, n in enumerate(nodes_sorted):
            qi = q[i]
            qn = q[i + 1] if i < len(nodes_sorted) - 1 else 0.0
            take_lps = max(qi - qn, 0.0)
            d_m3s = (take_lps / 1000.0) if st[i] == "open" else 0.0
            set_junction_demand(wn.get_node(n), d_m3s)

        # 5.2 “只到最远取水点有过流”：找最远 open 节点索引，并关闭其下游管段
        open_idx = [i for i, s in enumerate(st) if s == "open" and q[i] > flow_eps_lps]
        if not open_idx:
            # 没有取水：关整条支管（或你也可以只关进口）
            for a, b in zip(nodes_sorted[:-1], nodes_sorted[1:]):
                pid = edge2pid.get(frozenset((a, b)))
                set_link_open(wn, pid, False)
        else:
            last = max(open_idx)

            # last 之后的段全部关闭
            for j in range(last, len(nodes_sorted) - 1):
                a, b = nodes_sorted[j], nodes_sorted[j + 1]
                pid = edge2pid.get(frozenset((a, b)))
                set_link_open(wn, pid, False)

            # last 之前的段确保打开（防止上一个轮灌组残留关闭）
            for j in range(0, last):
                a, b = nodes_sorted[j], nodes_sorted[j + 1]
                pid = edge2pid.get(frozenset((a, b)))
                set_link_open(wn, pid, True)

    # 6) 自检：打印本组真正被赋了非零需求的节点（即“取水点”）
    if debug:
        nz = []
        for n in wn.junction_name_list:
            j = wn.get_node(n)
            d = j.demand_timeseries_list[0].base_value if len(j.demand_timeseries_list) > 0 else 0.0
            if abs(d) > 1e-12:
                nz.append((n, float(d)))
        print("Nonzero demands (m3/s):", nz)


# ============================================================
# 7) 用“路径并集”定义 active links/nodes（避免流量阈值噪声）
# ============================================================
def active_subnet_from_demands(wn: wntr.network.WaterNetworkModel,
                              source_node: str = "J0",
                              demand_eps_m3s: float = 1e-12):
    """
    active = 从 source 到所有 demand>0 的节点的最短路径并集（仅在 OPEN 管段上找路径）
    """
    demand_nodes = []
    for n in wn.junction_name_list:
        j = wn.get_node(n)
        d = j.demand_timeseries_list[0].base_value if len(j.demand_timeseries_list) > 0 else 0.0
        if d > demand_eps_m3s:
            demand_nodes.append(n)

    # 构图：只包含 OPEN 的管段
    G = nx.Graph()
    for lid, link in wn.links():
        # OPEN/CLOSED 的判定尽量兼容
        st = None
        if hasattr(link, "initial_status"):
            st = str(link.initial_status).upper()
        elif hasattr(link, "status"):
            st = str(link.status).upper()

        if st and "CLOSED" in st:
            continue

        u = link.start_node_name
        v = link.end_node_name
        w = getattr(link, "length", 1.0)
        G.add_edge(u, v, link_id=lid, weight=w)

    active_links = set()
    active_nodes = set([source_node])

    for nid in demand_nodes:
        if nid not in G:
            continue
        path = nx.shortest_path(G, source_node, nid, weight="weight")
        active_nodes.update(path)
        for a, b in zip(path[:-1], path[1:]):
            active_links.add(G[a][b]["link_id"])

    return sorted(active_nodes), sorted(active_links)


# ============================================================
# 8) 稳态计算：输入 group_df（而不是“节点列表”）
# ============================================================
edge2pid = build_edge_to_pipe_id(pipes_df)

def run_steady_active_pressures(wn_base: wntr.network.WaterNetworkModel,
                               group_df: pd.DataFrame,
                               source_node: str = "J0",
                               debug: bool = False):
    wn = copy.deepcopy(wn_base)

    apply_group_boundary(wn, group_df, edge2pid, debug=debug)

    sim = wntr.sim.EpanetSimulator(wn)
    res = sim.run_sim()

    p_t0 = res.node["pressure"].iloc[0]      # 节点压力（m）
    q_t0 = res.link["flowrate"].iloc[0]      # 管段流量（m3/s）

    active_nodes, active_links = active_subnet_from_demands(wn, source_node=source_node)
    p_active = p_t0.loc[active_nodes] if len(active_nodes) else p_t0

    return p_active, active_links, q_t0


# ============================================================
# 9) 打印 30 组简要信息（方便你核对）
# ============================================================
def group_name(i): return f"group{i}"

print("\n====================")
print("All 30 irrigation groups (summary)")
print("====================\n")

for i in range(1, 31):
    gname = group_name(i)
    if gname not in groups_dict:
        print(f"[{gname}] !!! sheet not found")
        continue

    g = parse_group_sheet(groups_dict[gname])

    # 支管 open（取水）节点
    take_nodes = g[(g["node"].str.match(r"^J\d+$")) & (g["state"] == "open") & (g["node"].apply(_node_num) >= 10)]["node"].tolist()

    # 干管 open/close
    trunk_states = g[(g["node"].str.match(r"^J\d+$")) & (g["node"].apply(_node_num) < 10)][["node", "state"]].values.tolist()

    print(f"[{gname}] trunks={trunk_states} take(open)={take_nodes}")


# ============================================================
# 10) 示例：group1 vs group2
# ============================================================
p1, links1, q1 = run_steady_active_pressures(wn_base, groups_dict["group1"], debug=True)
p2, links2, q2 = run_steady_active_pressures(wn_base, groups_dict["group2"], debug=True)

print("\n--- group1 ---")
print("active links:", links1)
print("active node pressures (m):\n", p1)

print("\n--- group2 ---")
print("active links:", links2)
print("active node pressures (m):\n", p2)


# ============================================================
# 11) 批量：30 组 -> 29 次切换（before/after）
# ============================================================
switch_results = []

for i in range(1, 30):
    before_name = group_name(i)
    after_name = group_name(i + 1)

    pB, linksB, _ = run_steady_active_pressures(wn_base, groups_dict[before_name])
    pA, linksA, _ = run_steady_active_pressures(wn_base, groups_dict[after_name])

    switch_results.append({
        "switch": f"{before_name}->{after_name}",
        "before_active_links": linksB,
        "before_nodes": list(pB.index),
        "before_press_m": [float(x) for x in pB.values],
        "after_active_links": linksA,
        "after_nodes": list(pA.index),
        "after_press_m": [float(x) for x in pA.values],
    })

# 需要的话写文件（给 TSNet/GNN 用）
import json
out_path = Path("switch_results.json")
out_path.write_text(json.dumps(switch_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved: {out_path.resolve()}")
