import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform

# --- 关键修改 1：解决中文乱码 ---
# 自动检测系统并设置字体（适配 Windows/Mac）
import platform

if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == "Darwin":  # MacOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置页面配置
st.set_page_config(page_title="CLT 模拟器", layout="wide")

st.title("📊 中心极限定理 (CLT) 交互式仿真平台")
st.markdown("该系统展示了**独立同分布随机变量序列**的均值，在样本容量较大时，其分布趋于**正态分布**的过程。")

# --- 2. 参数输入模块（专利：多源分布参数调节机构） ---
st.sidebar.header("🔧 配置模拟参数")

# 丰富的分布选择列表
dist_list = [
    "0-1 分布 (Bernoulli)",
    "二项分布 (Binomial)",
    "几何分布 (Geometric)",
    "均匀分布 (Uniform)",
    "指数分布 (Exponential)",
    "正态分布 (Normal)",
    "卡方分布 (Chi-Square)",
    "t 分布",
    "F 分布",
    "泊松分布 (Poisson)"
]

dist_type = st.sidebar.selectbox("选择母体分布类型", dist_list)

# 动态参数调节：根据不同的分布显示对应的参数滑块
st.sidebar.subheader("母体分布自身参数")
if dist_type == "0-1 分布 (Bernoulli)":
    p_param = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5)
elif dist_type == "二项分布 (Binomial)":
    n_binom = st.sidebar.slider("试验次数 n_trial", 1, 50, 10)
    p_binom = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5)
elif dist_type == "卡方分布 (Chi-Square)":
    df_chi = st.sidebar.slider("自由度 df", 1, 20, 5)
elif dist_type == "t 分布":
    df_t = st.sidebar.slider("自由度 df", 1, 50, 10)
elif dist_type == "F 分布":
    df_n = st.sidebar.slider("分子自由度 dfn", 1, 50, 10)
    df_d = st.sidebar.slider("分母自由度 dfd", 1, 50, 20)

# 核心抽样参数
st.sidebar.subheader("CLT 抽样参数")
n = st.sidebar.slider("样本容量 (n): 每次抽取的样本数", min_value=1, max_value=5000, value=30)
N = st.sidebar.slider("模拟次数 (N): 重复抽样的总次数", min_value=100, max_value=10000, value=2000)


# --- 3. 核心计算模块（专利：数据矩阵处理算法） ---
def generate_means(dist_type, n, N):
    if dist_type == "0-1 分布 (Bernoulli)":
        data = bernoulli.rvs(p_param, size=(N, n))
    elif dist_type == "二项分布 (Binomial)":
        data = binom.rvs(n_binom, p_binom, size=(N, n))
    elif dist_type == "几何分布 (Geometric)":
        data = geom.rvs(0.5, size=(N, n))
    elif dist_type == "均匀分布 (Uniform)":
        data = uniform.rvs(size=(N, n))
    elif dist_type == "指数分布 (Exponential)":
        data = expon.rvs(size=(N, n))
    elif dist_type == "正态分布 (Normal)":
        data = norm.rvs(loc=0, scale=1, size=(N, n))
    elif dist_type == "卡方分布 (Chi-Square)":
        data = chi2.rvs(df_chi, size=(N, n))
    elif dist_type == "t 分布":
        data = t.rvs(df_t, size=(N, n))
    elif dist_type == "F 分布":
        data = f.rvs(df_n, df_d, size=(N, n))
    else:  # Poisson
        data = poisson.rvs(mu=3, size=(N, n))

    return np.mean(data, axis=1)


sample_means = generate_means(dist_type, n, N)

# --- 4. 可视化渲染模块 ---
fig, ax = plt.subplots(figsize=(10, 5))

# 绘制直方图
ax.hist(sample_means, bins=50, density=True, alpha=0.6, color='#1f77b4', label='样本均值经验分布')

# 拟合正态曲线（理论值线）
mu_fit, std_fit = norm.fit(sample_means)
x = np.linspace(min(sample_means), max(sample_means), 100)
p = norm.pdf(x, mu_fit, std_fit)
ax.plot(x, p, 'r--', linewidth=2, label='拟合正态曲线')

ax.set_title(f"{dist_type} 在样本容量 n={n} 时的均值收敛演示", fontsize=14)
ax.set_xlabel("样本均值数值")
ax.set_ylabel("概率密度")
ax.legend()

st.pyplot(fig)

# --- 5. 统计指标显示 ---
st.subheader("📊 模拟结果统计")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("样本均值期望 (Mean)", f"{mu_fit:.4f}")
with c2:
    st.metric("样本均值标准差 (Std)", f"{std_fit:.4f}")
with c3:
    # 偏度计算，衡量正态性
    from scipy.stats import skew

    sk = skew(sample_means)
    st.metric("分布偏度 (Skewness)", f"{sk:.4f}")

st.info(

    "💡 专利提示：注意观察！随着 n 的增加（特别是到 5000 时），无论原始分布多么怪异（如 F 分布），均值分布都会变得非常对称且符合红色虚线。")
