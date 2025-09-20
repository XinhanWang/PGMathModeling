"""
轻量级图绘制工具（借助 networkx 和 matplotlib）
对应原 MATLAB 示例：数字标号图、字符串标号图、邻接矩阵绘制（带权重）
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
from matplotlib import font_manager as fm
import matplotlib

# 避免负号显示成方块/问号
matplotlib.rcParams['axes.unicode_minus'] = False

# 直接尝试将全局字体设置为 /home/xhwang/msyh.ttc（若存在）
_font_path_global = '/home/xhwang/msyh.ttc'
if os.path.exists(_font_path_global):
    try:
        # 注册字体文件到 matplotlib 字体管理器
        fm.fontManager.addfont(_font_path_global)
        # 通过 FontProperties 获取字体族名并设置为全局字体
        _fp = FontProperties(fname=_font_path_global)
        _font_name = _fp.get_name()
        if _font_name:
            matplotlib.rcParams['font.family'] = _font_name
            matplotlib.rcParams['font.sans-serif'] = [_font_name]
    except Exception:
        # 静默回退，不影响后续逻辑
        pass

def get_chinese_font(preferred_path=None):
    """
    返回一个可用的中文 FontProperties。
    尝试一系列常见路径与字体家族名，找不到时退回默认 FontProperties（不会抛异常）。
    """
    candidates = []
    # 优先使用用户指定的或全局设置的字体路径
    if preferred_path:
        candidates.append(preferred_path)
    candidates.append(_font_path_global)
    candidates.extend([
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK SC',
        'DejaVu Sans'
    ])
    for c in candidates:
        if not c:
            continue
        # 如果是文件路径且存在，直接使用
        try:
            if os.path.exists(c):
                return FontProperties(fname=c)
        except Exception:
            pass
        # 否则尝试当作字体家族名，让 font_manager 寻找
        try:
            fp = fm.findfont(FontProperties(family=c))
            if fp and os.path.exists(fp):
                return FontProperties(fname=fp)
        except Exception:
            pass
    # 找不到合适的中文字体则退回默认（不会抛错，但可能无法完美显示中文）
    return FontProperties()

def _contains_nonascii(labels):
    """
    检查标签列表中是否包含非 ASCII 字符（用于判断是否需要使用中文字体）。
    """
    return any(any(ord(ch) > 127 for ch in str(lbl)) for lbl in labels)

def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def _format_edge_labels(edge_labels):
    """整数权重不带小数显示。"""
    formatted = {}
    for k, v in edge_labels.items():
        if isinstance(v, (int, float)) and float(v).is_integer():
            formatted[k] = int(v)
        else:
            formatted[k] = v
    return formatted

def _draw_edge_labels_with_font(ax, pos, edge_labels, font):
    """
    在给定 ax 与 pos 上，使用指定 font (FontProperties) 手工绘制边标签，
    位置在边的中点处。edge_labels 为 {(u,v): label, ...}。
    """
    for (u, v), lbl in edge_labels.items():
        # 某些图中节点可能为可哈希的字符串或数字
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(x, y, str(lbl), fontproperties=font, horizontalalignment='center', verticalalignment='center')

def _dag_layer_layout(G):
    """
    针对有向无环图的简单分层布局（自顶向下）。
    """
    layers = list(nx.topological_generations(G))
    pos = {}
    total_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        if not layer:
            continue
        xs = np.linspace(0, 1, len(layer) + 2)[1:-1]
        y = -(layer_idx)  # 自顶向下
        for x, node in zip(xs, layer):
            pos[node] = (x, y)
    # 归一化/平移
    if pos:
        xs_all = [p[0] for p in pos.values()]
        ys_all = [p[1] for p in pos.values()]
        x_min, x_max = min(xs_all), max(xs_all)
        y_min, y_max = min(ys_all), max(ys_all)
        dx = x_max - x_min if x_max > x_min else 1.0
        dy = y_max - y_min if y_max > y_min else 1.0
        for k, (x, y) in pos.items():
            pos[k] = ((x - x_min) / dx, (y - y_min) / dy)
    return pos

def _compute_layout(G, layout_seed=None, layout_name=None):
    """
    layout_name:
      None 或 'auto' 自动挑选
      'graphviz'/'dot'/'neato' 强制 graphviz
      'spring'/'kamada'/'kamada_kawai'/'circular'/'shell'
    自动策略:
      DAG -> 优先 dot -> 分层布局
      小而稀疏 (n<=12 或 (n<=40 且 density<0.15)) -> kamada_kawai
      稠密 (density>0.6) -> circular
      中等规模 40<n<=150 -> spring (调参)
      其他 -> spring
    """
    layout_key = (layout_name or 'auto').lower()

    # 直接指定的简单布局
    if layout_key in ('circular',):
        return nx.circular_layout(G)
    if layout_key in ('shell',):
        return nx.shell_layout(G)
    if layout_key in ('kamada', 'kamada_kawai'):
        return nx.kamada_kawai_layout(G)
    if layout_key in ('spring',):
        return nx.spring_layout(G, seed=layout_seed)

    # graphviz 强制
    if layout_key in ('graphviz', 'dot', 'neato'):
        progs = ['dot', 'neato'] if layout_key == 'graphviz' else [layout_key]
        for prog in progs:
            try:
                return nx.nx_agraph.graphviz_layout(G, prog=prog)
            except Exception:
                continue
        # 失败回退 spring
        return nx.spring_layout(G, seed=layout_seed)

    # 自动策略
    n = max(1, G.number_of_nodes())
    m = G.number_of_edges()
    if n <= 1:
        return {nxt: (0.0, 0.0) for nxt in G.nodes()}
    # 有向图密度基于 n*(n-1)，无向图基于 n*(n-1)/2
    if G.is_directed():
        density = m / (n * (n - 1) + 1e-9)
    else:
        density = (2 * m) / (n * (n - 1) + 1e-9)

    # DAG 优先
    if G.is_directed():
        if nx.is_directed_acyclic_graph(G):
            # 尝试 graphviz dot
            try:
                return nx.nx_agraph.graphviz_layout(G, prog='dot')
            except Exception:
                return _dag_layer_layout(G)


    # 小稀疏
    if (n <= 12) or (n <= 40 and density < 0.15):
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass

    # 稠密
    if density > 0.6 and n <= 120:
        return nx.circular_layout(G)

    # 中等
    if 40 < n <= 150:
        k = 0.8 / np.sqrt(n)
        k = float(np.clip(k, 0.03, 0.5))
        return nx.spring_layout(G, seed=layout_seed, k=k, iterations=300)

    # 默认
    k = 0.6 / np.sqrt(n)
    k = float(np.clip(k, 0.04, 0.9))
    try:
        return nx.spring_layout(G, seed=layout_seed, k=k, iterations=250)
    except Exception:
        return nx.spring_layout(G, seed=layout_seed)

def plot_numeric_weighted(s, t, w, ax=None, title=None, layout_seed=None):
    """
    s, t: iterable of integer node indices (1-based or 0-based)
    w: iterable of edge weights
    """
    if ax is None:
        fig, ax = plt.subplots()
    # 统一判断 s 与 t 是否为 1-based
    s_arr = np.array(s)
    t_arr = np.array(t)
    if min(s_arr.min(), t_arr.min()) >= 1:
        s = (s_arr - 1).tolist()
        t = (t_arr - 1).tolist()
    G = nx.Graph()
    edges = [(int(u), int(v), float(ww)) for u, v, ww in zip(s, t, w)]
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw(G, pos, with_labels=True, ax=ax, node_size=500, linewidths=2)
    edge_labels = _format_edge_labels(nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    if title:
        ax.set_title(title)
    _clean_axes(ax)
    return ax

def plot_numeric_unweighted(s, t, ax=None, title=None, layout_seed=None):
    if ax is None:
        fig, ax = plt.subplots()
    s_arr = np.array(s)
    t_arr = np.array(t)
    if min(s_arr.min(), t_arr.min()) >= 1:
        s = (s_arr - 1).tolist()
        t = (t_arr - 1).tolist()
    G = nx.Graph()
    G.add_edges_from([(int(u), int(v)) for u, v in zip(s, t)])
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw(G, pos, with_labels=True, ax=ax, node_size=500, linewidths=2)
    if title:
        ax.set_title(title)
    _clean_axes(ax)
    return ax

def plot_string_weighted(s, t, w, ax=None, title=None, layout_seed=None, layout='auto'):
    """
    新增参数:
      layout: 'auto' | 'spring' | 'kamada' | 'graphviz' | 'circular' | 'shell' | 'dot' | 'neato'
    """
    if ax is None:
        fig, ax = plt.subplots()
    G = nx.Graph()
    edges = [(u, v, float(ww)) for u, v, ww in zip(s, t, w)]
    G.add_weighted_edges_from(edges)
    pos = _compute_layout(G, layout_seed=layout_seed, layout_name=layout)
    n_nodes = max(1, G.number_of_nodes())
    node_size = int(np.clip(600 * (8 / (n_nodes + 7)), 120, 900))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2)
    font = get_chinese_font()
    for n, (x, y) in pos.items():
        ax.text(x, y, str(n), fontproperties=font, horizontalalignment='center', verticalalignment='center')
    edge_labels = _format_edge_labels(nx.get_edge_attributes(G, 'weight'))
    _draw_edge_labels_with_font(ax, pos, edge_labels, font)
    if title:
        ax.set_title(title)
    _clean_axes(ax)
    return ax

def plot_from_adjacency(adj, labels=None, ax=None, title=None, layout_seed=None, layout='auto'):
    """
    新增参数 layout 见 plot_string_weighted。
    """
    adj = np.array(adj)
    n = adj.shape[0]
    if labels is None:
        labels = list(range(1, n+1))  # 使用 1..n 作为标签，便于与 MATLAB 类比
    # 判断是否对称
    directed = not np.allclose(adj, adj.T)
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(labels)  # 确保孤立节点出现
    if directed:
        for i in range(n):
            for j in range(n):
                w = adj[i, j]
                if w != 0:
                    G.add_edge(labels[i], labels[j], weight=float(w))
    else:
        for i in range(n):
            for j in range(i+1, n):
                w = adj[i, j]
                if w != 0:
                    G.add_edge(labels[i], labels[j], weight=float(w))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    pos = _compute_layout(G, layout_seed=layout_seed, layout_name=layout)
    n_nodes = max(1, G.number_of_nodes())
    node_size = int(np.clip(600 * (8 / (n_nodes + 7)), 120, 900))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2)
    if _contains_nonascii(labels):
        font = get_chinese_font()
        for n_, (x, y) in pos.items():
            ax.text(x, y, str(n_), fontproperties=font, horizontalalignment='center', verticalalignment='center')
        edge_labels = _format_edge_labels(nx.get_edge_attributes(G, 'weight'))
        _draw_edge_labels_with_font(ax, pos, edge_labels, font)
    else:
        nx.draw_networkx_labels(G, pos, ax=ax)
        edge_labels = _format_edge_labels(nx.get_edge_attributes(G, 'weight'))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    if title:
        ax.set_title(title)
    _clean_axes(ax)
    return ax

if __name__ == '__main__':
    # 演示：复现原 MATLAB 示例
    # 数字标号示例（带权）
    s = [1, 2, 3]
    t = [4, 1, 2]
    w = [5, 2, 6]
    plot_numeric_weighted(s, t, w, title='数字标号带权示例')
    plt.show()

    # 数字标号示例（不带权）
    plot_numeric_unweighted(s, t, title='数字标号不带权示例')
    plt.show()

    # 字符串标号示例（中文）
    s_str = ['北京', '上海', '广州', '深圳', '上海']
    t_str = ['上海', '广州', '深圳', '北京', '深圳']
    w_str = [10, 65, 3, 90, 60]
    plot_string_weighted(s_str, t_str, w_str, title='字符串标号带权示例', layout='circular')
    plt.show()

    # 邻接矩阵示例
    c = np.array([
        [0,15,10,20,0,0,0,0],
        [0,0,0,0,7,10,0,0],
        [0,0,0,0,0,8,2,0],
        [0,0,0,0,0,0,18,0],
        [0,0,0,0,0,0,0,6],
        [0,0,0,0,0,0,0,16],
        [0,0,0,0,0,0,0,20],
        [0,0,0,0,0,0,0,0],
    ])
    labels = [f'V{i+1}' for i in range(c.shape[0])]
    plot_from_adjacency(c, labels=labels, title='邻接矩阵示例（自动布局）', layout='auto')
    plt.show()
