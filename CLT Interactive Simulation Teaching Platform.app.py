import matplotlib
matplotlib.use('agg')  # 设置为 agg 后端，用于无头环境（如 Streamlit 和其他云平台）

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# 补充导入伽玛分布 + 峰度函数kurtosis
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform, skew, gamma, kurtosis
import os
import matplotlib.font_manager as fm
import requests

# ===================== 优化：修复路径问题 + 强化中文字体配置 =====================
def setup_chinese_font():
    """统一配置中文字体，优先加载本地字体，无则使用系统字体，兼容本地+Streamlit Cloud"""
    # 下载并加载字体文件（通过 GitHub URL）
    font_url = "https://github.com/fenggeHan/CLT-Interactive-Simulation-Teaching-Platform/raw/main/simhei.ttf"
    # 安全兼容所有环境的路径：当前脚本所在目录 + fonts 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current_dir, "fonts")
    font_path = os.path.join(font_dir, "simhei.ttf")

    # 如果本地字体文件不存在，则从 GitHub 下载
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

    # 加载字体
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
    page_title="中心极限定理 (CLT) 交互式仿真平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 需求1：缩小标题字体（用markdown自定义字体大小，比默认title小）
st.markdown(
    '<h1 style="font-size:28px; margin-bottom:20px;">📊 中心极限定理 (CLT) 交互式仿真平台</h1>',
    unsafe_allow_html=True
)
# 替代原有 st.title("#######📊 中心极限定理 (CLT) 交互式仿真平台")

st.markdown("""
该系统展示了**独立同分布随机变量序列**的均值，在样本容量较大时，其分布趋于**正态分布**的过程。
支持多种母体分布类型，可动态调节参数观察收敛效果。
""")

# ===================== 侧边栏参数配置 =====================
st.sidebar.header("🔧 配置模拟参数")

# 需求1：添加伽玛分布，放在泊松分布后面
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
    "伽玛分布 (Gamma)"  # 新增：伽玛分布
]
dist_type = st.sidebar.selectbox("选择母体分布类型", dist_list)

# 初始化所有可能的分布参数（包含新增参数，避免未定义报错）
p_param = 0.5
n_binom = 10
p_binom = 0.5
p_geom = 0.5
mu_pois = 3
# 需求2：初始化正态分布、指数分布可调节参数
norm_loc = 0    # 正态分布均值
norm_scale = 1  # 正态分布标准差
expon_scale = 1 # 指数分布尺度参数（对应均值=scale）
# 初始化伽玛分布参数
gamma_a = 2     # 伽玛分布形状参数
gamma_scale = 1 # 伽玛分布尺度参数
df_chi = 5
df_t = 10
df_n = 10
df_d = 20

# 动态参数调节（每个分支都定义参数，避免变量未定义）
st.sidebar.subheader("母体分布自身参数")
if dist_type == "0-1 分布 (Bernoulli)":
    p_param = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "二项分布 (Binomial)":
    n_binom = st.sidebar.slider("试验次数 n_trial", 1, 50, 10, step=1)
    p_binom = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "几何分布 (Geometric)":
    p_geom = st.sidebar.slider("成功概率 p", 0.1, 0.9, 0.5, step=0.05)
elif dist_type == "均匀分布 (Uniform)":
    # 均匀分布可保留默认，也可扩展，此处保持原有逻辑
    pass
elif dist_type == "指数分布 (Exponential)":
    # 需求2：添加指数分布可调节参数（尺度参数，均值=scale）
    expon_scale = st.sidebar.slider("尺度参数 scale（均值=scale）", 0.1, 10.0, 1.0, step=0.1)
elif dist_type == "正态分布 (Normal)":
    # 需求2：添加正态分布可调节参数（均值loc、标准差scale）
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
    # 新增：伽玛分布参数调节
    gamma_a = st.sidebar.slider("形状参数 a", 0.5, 20.0, 2.0, step=0.5)
    gamma_scale = st.sidebar.slider("尺度参数 scale", 0.1, 10.0, 1.0, step=0.1)

# 需求3：样本容量滑动条标注教学常用范围及临界值
st.sidebar.subheader("CLT 抽样参数")
n = st.sidebar.slider(
    "样本容量 (n)：每次抽取的样本数【教学常用：30(大样本临界值)、100、500】",
    min_value=1,
    max_value=5000,
    value=30,
    step=10,
    help="教学关键临界值：n=30（大样本最低要求）、n=100（收敛效果明显）、n=500（收敛效果极佳）"
)
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
            # 使用可调节的指数分布参数
            data = expon.rvs(scale=expon_scale, size=(N, n))
        elif dist_type == "正态分布 (Normal)":
            # 使用可调节的正态分布参数
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
            # 新增：伽玛分布数据生成
            data = gamma.rvs(gamma_a, scale=gamma_scale, size=(N, n))
        else:
            data = norm.rvs(loc=0, scale=1, size=(N, n))
        
        # 计算每行（每次抽样）的均值
        sample_means = np.mean(data, axis=1)
        return sample_means
    
    except Exception as e:
        st.error(f"数据生成出错：{str(e)}")
        return np.array([])

# 生成样本均值
sample_means = generate_means(dist_type, n, N)

# ===================== 可视化模块（中文正常显示） =====================
if len(sample_means) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图（中文label正常显示）
    ax.hist(
        sample_means, 
        bins=min(50, len(sample_means)//50),
        density=True, 
        alpha=0.7, 
        color='#2E86AB', 
        edgecolor='white',
        label='样本均值经验分布'
    )

    # 拟合正态曲线（中文label正常显示）
    mu_fit, std_fit = norm.fit(sample_means)
    x = np.linspace(min(sample_means), max(sample_means), 200)
    p = norm.pdf(x, mu_fit, std_fit)
    ax.plot(x, p, 'r--', linewidth=2.5, label='拟合正态曲线')

    # 显式获取中文字体（双重保障）
    try:
        font_prop = fm.FontProperties(fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "simhei.ttf"), size=11)
    except:
        font_prop = fm.FontProperties(family=['SimHei', 'WenQuanYi Zen Hei'], size=11)

    # 设置标题（dist_type中文正常显示，包含伽玛分布）
    ax.set_title(
        f"{dist_type} 在样本容量 n={n} 时的均值收敛演示",
        fontsize=16, fontweight='bold', fontproperties=font_prop
    )
    ax.set_xlabel("样本均值数值", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("概率密度", fontsize=12, fontproperties=font_prop)
    
    # 图例中文正常显示
    ax.legend(prop=font_prop, fontsize=11)
    ax.grid(alpha=0.3)

    # 显示图表
    st.pyplot(fig)

        # ===================== 统计指标展示 =====================
    st.subheader("📊 模拟结果统计")
    # 计算偏度和峰度
    sk = skew(sample_means)
    kurt = kurtosis(sample_means)
    # 判断偏度颜色区间
    abs_sk = abs(sk)
    if abs_sk < 0.5:
        skewness_color = "#2ecc71"  # 绿色
    elif 0.5 <= abs_sk <= 1:
        skewness_color = "#f1c40f"  # 黄色
    else:
        skewness_color = "#e74c3c"  # 红色

    # 5列布局，统一原生metric样式
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("期望 (Mean)", f"{mu_fit:.4f}")
    with c2:
        st.metric("标准差 (Std)", f"{std_fit:.4f}")
    with c3:
        # 模仿原生st.metric样式 + 调小数字尺寸
        st.markdown(f"""
        <div style="background-color: var(--st-card-bg-color); padding: 1rem; border-radius: 0.5rem; height: 100%;">
            <div style="font-size: 14px; color: var(--st-text-secondary-color); margin-bottom: 0.25rem;">分布偏度 (Skewness)</div>
            <div style="font-size: 20px; font-weight: 600; color: {skewness_color};">{sk:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.metric("分布峰度 (Kurtosis)", f"{kurt:.4f}")
    with c5:
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
""")  # 需求3：修改使用说明语句




