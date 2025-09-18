import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

def get_chinese_font():
    """
    返回指定路径的微软雅黑字体 FontProperties 实例。
    """
    return FontProperties(fname='/home/xhwang/msyh.ttc')

def pca_from_scratch(X, n_components):
    """
    简单的 PCA 实现（基于协方差矩阵的特征分解）。
    参数:
      X: shape (n_samples, n_features)
      n_components: 降维后的维数
    返回:
      X_proj: 投影后的数据 (n_samples, n_components)
      components: 主成分向量 (n_components, n_features)
      explained_variance_ratio: 每个主成分解释的方差比
    """
    # 中心化
    X_centered = X - np.mean(X, axis=0)
    # 协方差矩阵 (features x features)
    cov = np.cov(X_centered, rowvar=False)
    # 特征值与特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)  # 对称矩阵使用 eigh 更稳定
    # 按特征值降序排序
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # 取前 n_components
    components = eigvecs[:, :n_components].T  # (n_components, n_features)
    # 投影
    X_proj = np.dot(X_centered, components.T)
    # 解释的方差比
    explained_variance_ratio = eigvals[:n_components] / np.sum(eigvals)
    return X_proj, components, explained_variance_ratio

def demo(seed=42):
    """
    生成合成数据并演示 PCA（自实现 vs sklearn）。
    """
    font_prop = get_chinese_font()
    print("已指定中文字体: /home/xhwang/msyh.ttc")

    try:
        from sklearn.decomposition import PCA as SKPCA
    except Exception:
        SKPCA = None

    np.random.seed(seed)
    # 生成三维相关数据（便于可视化）
    n = 200
    t = np.linspace(0, 4 * np.pi, n)
    x1 = np.cos(t) + 0.1 * np.random.randn(n)
    x2 = 0.5 * np.sin(t) + 0.1 * np.random.randn(n)
    x3 = 0.3 * np.cos(t) + 0.05 * np.random.randn(n)
    X = np.vstack([x1, x2, x3]).T  # (n, 3)

    # 使用自实现 PCA 降到 2 维
    X2, components, var_ratio = pca_from_scratch(X, 2)
    print("自实现 PCA - 解释方差比:", np.round(var_ratio, 4))

    # 使用 sklearn 的 PCA（如果可用）
    if SKPCA is not None:
        skp = SKPCA(n_components=2)
        X2_sk = skp.fit_transform(X)
        print("sklearn PCA - 解释方差比:", np.round(skp.explained_variance_ratio_, 4))
    else:
        X2_sk = None
        print("sklearn 未安装或不可用，跳过 sklearn 演示。")

    # 可视化：原始 3D（投影到 3D 散点）与降维后 2D
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis', s=15)
    ax1.set_title('原始 3D 数据', fontproperties=font_prop)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(X2[:, 0], X2[:, 1], c=t, cmap='viridis', s=15)
    ax2.set_title('自实现 PCA 投影 (2D)', fontproperties=font_prop)
    ax2.set_xlabel('PC1', fontproperties=font_prop)
    ax2.set_ylabel('PC2', fontproperties=font_prop)

    ax3 = fig.add_subplot(1, 3, 3)
    if X2_sk is not None:
        ax3.scatter(X2_sk[:, 0], X2_sk[:, 1], c=t, cmap='viridis', s=15)
        ax3.set_title('sklearn PCA 投影 (2D)', fontproperties=font_prop)
    else:
        ax3.text(0.5, 0.5, 'sklearn 未安装', ha='center', fontsize=12, fontproperties=font_prop)
        ax3.set_title('sklearn PCA (不可用)', fontproperties=font_prop)
    ax3.set_xlabel('PC1', fontproperties=font_prop)
    ax3.set_ylabel('PC2', fontproperties=font_prop)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo()
