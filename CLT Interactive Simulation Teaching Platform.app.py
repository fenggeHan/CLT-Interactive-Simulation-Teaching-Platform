import matplotlib
matplotlib.use('agg')  # 设置为 agg 后端，用于无头环境（如 Streamlit 和其他云平台）

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform, skew, gamma
import os
import matplotlib.font_manager as fm
import requests

# ===================== 优化：修复路径问题 + 强化中文字体配置 =====================
def setup_chinese_font():
    """统一配置中文字体，优先加载本地字体，无则使用系统字体，兼容本地+Streamlit Cloud"""
    font_url = "https://github.com/fenggeHan/CLT-Interactive-Simulation-Teaching-Platform/raw/main/simhei.ttf"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current_dir, "fonts")
    font_path = os.path.join(font_dir, "simhei.ttf")

    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        try:
            response = requests.get(font_url, timeout=15)
            response.raise_for_status()
            with open(font_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            st.warning(f"下载字体失败，将使用系统默认中文字体：{str(e)}")
            plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams["axes.unicode_minus"] = False
            return

    try:
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        st.warning(f"加载字体失败，将使用系统默认中文字体：{str(e)}")
        plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams["axes.unicode_minus"] = False

# 执行字体配置
setup_chinese_font()

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="中心极限定理交互式仿真平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 中心极限定理 (CLT) 交互式仿真平台")
st.markdown("""
该系统展示了**独立同分布随机变量序列**的均值，在样本容量较大时，其分布趋于**正态分布**的过程。
支持多种母体分布类型，可动态调节参数观察收敛效果。
""")

# ===================== 侧边栏参数配置 =====================
st.sidebar.header("🔧 配置模拟参数")

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
    "泊松分布 (Poisson)",
    "伽玛分布 (Gamma)"
]
dist_type = st.sidebar.selectbox("选择母体分布类型", dist_list)

# 初始化所有分布参数
p_param = 0.5
n_binom = 10
p_binom = 0.5
p_geom = 0.5
mu_pois = 3
norm_loc = 0    
norm_scale = 1  
expon_scale = 1 
gamma_a = 2     
gamma_scale = 1 
df_chi = 5
df_t = 10
df_n = 10
df_d = 20

# 母体分布自身参数
st.sidebar.subheader("母体分布自身参数")
if dist_type == "0-1 分布 (Bernoulli)":
    p_param = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "二项分布 (Binomial)":
    n_binom = st.sidebar.slider("试验次数 n_trial", 1, 50, 10, step=1)
    p_binom = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "几何分布 (Geometric)":
    p_geom = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "指数分布 (Exponential)":
    expon_scale = st.sidebar.slider("尺度参数 scale（均值=scale）", 0.1, 10.0, 1.0, step=0.1)
elif dist_type == "正态分布 (Normal)":
    norm_loc = st.sidebar.slider("均值 μ (loc)", -10.0, 10.0, 0.0, step=0.5)
    norm_scale = st.sidebar.slider("标准差 σ (scale)", 0.1, 10.0, 1.0, step=0.1)
elif dist_type == "卡方分布 (Chi-Square)":
    df_chi = st.sidebar.slider("自由度 df", 1, 20, 5, step=1)
elif dist_type == "t 分布":
    df_t = st.sidebar.slider("自由度 df", 1, 50, 10, step=1)
elif dist_type == "F 分布":
    df_n = st.sidebar.slider("分子自由度 dfn", 1, 50, 10, step=1)
    df_d = st.sidebar.slider("分母自由度 dfd", 1, 50, 20, step=1)
elif dist_type == "泊松分布 (Poisson)":
    mu_pois = st.sidebar.slider("均值 μ", 1, 20, 3, step=1)
elif dist_type == "伽玛分布 (Gamma)":
    gamma_a = st.sidebar.slider("形状参数 a", 0.5, 20.0, 2.0, step=0.5)
    gamma_scale = st.sidebar.slider("尺度参数 scale", 0.1, 10.0, 1.0, step=0.1)

# ===================== 核心修改：滑动条直接标注临界值（30、100、500） =====================
st.sidebar.subheader("CLT 抽样参数")
# 用容器包裹滑动条 + 临界值标注
with st.container():
    # 样本容量滑动条（去掉文字说明，改为轨道标注）
    n = st.slider(
        "样本容量 (n)：每次抽取的样本数",
        min_value=1,
        max_value=5000,
        value=30,
        step=10,
        key="sample_size_slider"
    )
    # 通过CSS在滑动条上方对应位置添加红色临界值标注
    st.markdown("""
    <style>
    /* 标注容器：相对定位，对齐滑动条 */
    .slider-markers {
        position: relative;
        width: 100%;
        margin-top: -22px; /* 调整标注与滑动条的垂直距离 */
        height: 20px;
    }
    /* 标注样式：红色、加粗 */
    .slider-marker {
        position: absolute;
        font-size: 12px;
        color: #ff4b4b;
        font-weight: 700;
    }
    /* 计算每个临界值在滑动条上的相对位置：(目标值-最小值)/(最大值-最小值)*100% */
    .marker-30 { left: calc((30 - 1)/(5000 - 1) * 100%); }
    .marker-100 { left: calc((100 - 1)/(5000 - 1) * 100%); }
    .marker-500 { left: calc((500 - 1)/(5000 - 1) * 100%); }
    </style>
    <div class="slider-markers">
        <div class="slider-marker marker-30">30</div>
        <div class="slider-marker marker-100">100</div>
        <div class="slider-marker marker-500">500</div>
    </div>
    """, unsafe_allow_html=True)

# 模拟次数滑动条
N = st.sidebar.slider(
    "模拟次数 (N)：重复抽样的总次数",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100
)

# ===================== 核心计算函数 =====================
def generate_means(dist_type, n, N):
    """生成样本均值数组，增加参数校验，避免报错"""
    try:
        if dist_type == "0-1 分布 (Bernoulli)":
            data = bernoulli.rvs(p_param, size=(N, n))
        elif dist_type == "二项分布 (Binomial)":
            data = binom.rvs(n_binom, p_binom, size=(N, n))
        elif dist_type == "几何分布 (Geometric)":
            data = geom.rvs(p_geom, size=(N, n))
        elif dist_type == "均匀分布 (Uniform)":
            data = uniform.rvs(loc=0, scale=1, size=(N, n))
        elif dist_type == "指数分布 (Exponential)":
            data = expon.rvs(scale=expon_scale, size=(N, n))
        elif dist_type == "正态分布 (Normal)":
            data = norm.rvs(loc=norm_loc, scale=norm_scale, size=(N, n))
        elif dist_type == "卡方分布 (Chi-Square)":
            data = chi2.rvs(df_chi, size=(N, n))
        elif dist_type == "t 分布":
            data = t.rvs(df_t, size=(N, n))
        elif dist_type == "F 分布":
            data = f.rvs(df_n, df_d, size=(N, n))
        elif dist_type == "泊松分布 (Poisson)":
            data = poisson.rvs(mu_pois, size=(N, n))
        elif dist_type == "伽玛分布 (Gamma)":
            data = gamma.rvs(gamma_a, scale=gamma_scale, size=(N, n))
        else:
            data = norm.rvs(loc=0, scale=1, size=(N, n))
        
        sample_means = np.mean(data, axis=1)
        return sample_means
    
    except Exception as e:
        st.error(f"数据生成出错：{str(e)}")
        return np.array([])

# 生成样本均值
sample_means = generate_means(dist_type, n, N)

# ===================== 可视化模块 =====================
if len(sample_means) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        sample_means, 
        bins=min(50, len(sample_means)//50),
        density=True, 
        alpha=0.7, 
        color='#2E86AB', 
        edgecolor='white',
        label='样本均值经验分布'
    )

    mu_fit, std_fit = norm.fit(sample_means)
    x = np.linspace(min(sample_means), max(sample_means), 200)
    p = norm.pdf(x, mu_fit, std_fit)
    ax.plot(x, p, 'r--', linewidth=2.5, label='拟合正态曲线')

    try:
        font_prop = fm.FontProperties(fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "simhei.ttf"), size=11)
    except:
        font_prop = fm.FontProperties(family=['SimHei', 'WenQuanYi Zen Hei'], size=11)

    ax.set_title(
        f"{dist_type} 在样本容量 n={n} 时的均值收敛演示",
        fontsize=16, fontweight='bold', fontproperties=font_prop
    )
    ax.set_xlabel("样本均值数值", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("概率密度", fontsize=12, fontproperties=font_prop)
    
    ax.legend(prop=font_prop, fontsize=11)
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # ===================== 统计指标展示 =====================
    st.subheader("📊 模拟结果统计")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("样本均值期望 (Mean)", f"{mu_fit:.4f}")
    with c2:
        st.metric("样本均值标准差 (Std)", f"{std_fit:.4f}")
    with c3:
        sk = skew(sample_means)
        st.metric("分布偏度 (Skewness)", f"{sk:.4f}")
    with c4:
        normality = "✅ 接近正态" if abs(sk) < 0.5 else "❌ 偏离正态"
        st.metric("正态性判断", normality)

    st.info("""
    💡 核心规律：随着样本容量 n 的增加（尤其是≥30时），无论原始母体分布类型如何，
    样本均值的分布都会逐渐趋近于正态分布（红色虚线）；当 n≥1000 时，收敛效果会非常显著。
    """)
else:
    st.warning("⚠️ 数据生成失败，请检查参数设置或刷新页面重试")

# ===================== 底部说明 =====================
st.markdown("---")
st.markdown("""
### 📝 使用说明
1.  左侧可选择不同的母体分布类型，并调节对应参数；
2.  调整样本容量 n 和模拟次数 N，观察均值分布的收敛效果；
3.  偏度越接近0，说明分布越对称（越接近正态分布）。
""")
