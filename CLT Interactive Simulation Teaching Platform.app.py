import matplotlib
matplotlib.use('agg')  # 固定无头环境配置，避免动态切换
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli, binom, geom, chi2, t, f, poisson, expon, uniform, skew, gamma, kurtosis
import os
import matplotlib.font_manager as fm
import requests

# ===================== 新增：访问量统计功能（核心代码） =====================
def count_website_visits():
    """
    统计网页访问量，本地文本文件存储，避免重复计数（单会话仅计数1次）
    """
    # 兼容本地和Streamlit Cloud路径，存储访问量文件
    if 'STREAMLIT_SERVER_ROOT_PATH' in os.environ:
        count_file = "visit_count.txt"  # 云环境直接根目录
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        count_file = os.path.join(current_dir, "visit_count.txt")  # 本地环境绝对路径

    # 初始化：若文件不存在，创建并写入0
    if not os.path.exists(count_file):
        try:
            with open(count_file, "w", encoding="utf-8") as f:
                f.write("0")
        except Exception as e:
            st.warning(f"访问量统计文件初始化失败：{str(e)}")
            return "无法统计"

    # 用session_state标记是否已计数，避免Streamlit重运行重复累加
    if not hasattr(st.session_state, 'visit_counted'):
        st.session_state.visit_counted = False

    # 仅当未计数时，更新访问量
    if not st.session_state.visit_counted:
        try:
            # 读取当前访问量
            with open(count_file, "r", encoding="utf-8") as f:
                current_count = int(f.read().strip())
            # 累加1
            new_count = current_count + 1
            # 写回文件
            with open(count_file, "w", encoding="utf-8") as f:
                f.write(str(new_count))
            # 标记已计数，避免重复
            st.session_state.visit_counted = True
            return str(new_count)
        except Exception as e:
            st.warning(f"访问量更新失败：{str(e)}")
            return "无法统计"
    else:
        # 已计数时，直接读取当前访问量
        try:
            with open(count_file, "r", encoding="utf-8") as f:
                current_count = f.read().strip()
            return current_count
        except Exception as e:
            return "无法统计"

# 执行访问量统计，获取当前累计访问量
total_visits = count_website_visits()

# ===================== 优化：修复路径问题 + 强化中文字体配置 =====================
def setup_chinese_font():
    """统一配置中文字体，优先加载本地字体，无则使用系统字体"""
    font_url = "https://github.com/fenggeHan/CLT-Interactive-Simulation-Teaching-Platform/raw/main/simhei.ttf"
    # 兼容Streamlit Cloud的路径问题（避免__file__在云环境报错）
    if 'STREAMLIT_SERVER_ROOT_PATH' in os.environ:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    font_dir = os.path.join(current_dir, "fonts")
    font_path = os.path.join(font_dir, "simhei.ttf")

    # 全局字体配置标记，避免重复加载
    if not hasattr(st.session_state, 'font_setup_done'):
        st.session_state.font_setup_done = False

    if st.session_state.font_setup_done:
        return  # 已配置过字体，直接返回

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
            st.session_state.font_setup_done = True
            return

    try:
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams["axes.unicode_minus"] = False
        st.session_state.font_setup_done = True
    except Exception as e:
        st.warning(f"加载字体失败，将使用系统默认中文字体：{str(e)}")
        plt.rcParams['font.family'] = ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams["axes.unicode_minus"] = False
        st.session_state.font_setup_done = True

# 执行字体配置
setup_chinese_font()

# ===================== 页面基础配置 =====================
st.set_page_config(
    page_title="中心极限定理 (CLT) 交互式仿真平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    '<h1 style="font-size:32px; margin-bottom:20px;">📊 中心极限定理 (CLT) 交互式仿真平台</h1>',
    unsafe_allow_html=True
)

# 新增：在页面顶部展示访问量（也可移至侧边栏/底部）
#st.markdown(f'<p style="font-size:14px; color:#666;">📈 累计访问量：{total_visits}</p>', unsafe_allow_html=True)

st.markdown("""
该系统展示了**独立同分布随机变量序列**的均值，在样本容量较大时，其分布趋于**正态分布**的过程。
支持多种母体分布类型，可动态调节参数观察收敛效果。
""")

# ===================== 侧边栏参数配置 =====================
st.sidebar.header("🔧 配置模拟参数")
# 可选：也可在侧边栏展示访问量
st.sidebar.markdown(f"📈 累计访问量：{total_visits}")

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

# 母体分布自身参数调节
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

# CLT 抽样参数
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
    """生成样本均值数组，强化异常捕获和参数校验"""
    try:
        # 增加参数合法性校验
        if n <= 0 or N <= 0:
            st.error("样本容量n和模拟次数N必须为正整数")
            return np.array([])
        
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
        # 兜底返回空数组，避免动画直接中断
        return np.array([])

# 生成手动调节的样本均值
sample_means = generate_means(dist_type, n, N)

# ===================== 手动调节的可视化模块（前置） =====================
st.subheader("📈 手动调节结果可视化")
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

    ax.set_title(
        f"{dist_type} 在样本容量 n={n} 时的均值收敛演示",
        fontsize=16, fontweight='bold'
    )
    ax.set_xlabel("样本均值数值", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)  # 释放资源

    # ===================== 手动调节的统计指标展示 =====================
    st.subheader("📊 模拟结果统计")
    sk = skew(sample_means)
    kurt = kurtosis(sample_means)
    abs_sk = abs(sk)
    skewness_color = "#2ecc71" if abs_sk < 0.5 else "#f1c40f" if 0.5 <= abs_sk <= 1 else "#e74c3c"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("样本均值期望 (Mean)", f"{mu_fit:.4f}")
    with c2:
        st.metric("样本均值标准差 (Std)", f"{std_fit:.4f}")
    with c3:
        st.markdown(f"""
        <div style="background-color: var(--st-card-bg-color); padding: 1rem; border-radius: 0.5rem; height: 100%;">
            <div style="font-size: 14px; color: var(--st-text-secondary-color); margin-bottom: 0.25rem;">分布偏度 (Skewness)</div>
            <div style="font-size: 20px; font-weight: 600; color: {skewness_color};">{sk:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.metric("分布峰度 (Kurtosis)", f"{kurt:.4f}")
    with c5:
        normality = "✅ 接近正态" if abs_sk < 0.5 else "❌ 偏离正态"
        st.metric("正态性判断", normality)

    st.info("""
    💡 核心规律：随着样本容量 n 的增加（尤其是≥30时），无论原始母体分布类型如何，
    样本均值的分布都会逐渐趋近于正态分布（红色虚线）；当 n≥1000 时，收敛效果会非常显著。
    """)
else:
    st.warning("⚠️ 数据生成失败，请检查参数设置或刷新页面重试")

# ===================== 动画演示模块（后置，核心调整点） =====================
st.subheader("🎬 动画演示")
# 初始化动画状态标记（提前初始化，避免报错）
if 'anim_running' not in st.session_state:
    st.session_state.anim_running = False

# 拆分按钮逻辑，防止重复点击
col1, col2 = st.columns([1, 9])
with col1:
    animate_btn = st.button("开始动画演示", type="primary", disabled=st.session_state.anim_running)
    stop_btn = st.button("停止动画演示")

# 停止按钮逻辑
if stop_btn:
    st.session_state.anim_running = False
    st.rerun()  # 强制刷新页面，终止动画循环

# 创建占位符（在动画模块内初始化，对应新位置）
chart_placeholder = st.empty()
stats_placeholder = st.empty()

# 动画核心逻辑
if animate_btn:
    st.session_state.anim_running = True
    # 降低动画步长，减少资源占用（步长10，更快更流畅）
    for anim_n in range(1, 501, 10):
        # 检查是否停止
        if not st.session_state.anim_running:
            break
        
        anim_sample_means = generate_means(dist_type, anim_n, N)
        if len(anim_sample_means) == 0:
            with chart_placeholder:
                st.warning(f"n={anim_n}时数据生成失败，跳过该步")
            continue

        # 简化字体配置，使用全局已配置的字体
        fig, ax = plt.subplots(figsize=(12, 6))
        # 绘制直方图
        ax.hist(
            anim_sample_means, 
            bins=min(50, len(anim_sample_means)//50),
            density=True, 
            alpha=0.7, 
            color='#2E86AB', 
            edgecolor='white',
            label='样本均值经验分布'
        )
        # 拟合正态曲线
        mu_fit, std_fit = norm.fit(anim_sample_means)
        x = np.linspace(min(anim_sample_means), max(anim_sample_means), 200)
        p = norm.pdf(x, mu_fit, std_fit)
        ax.plot(x, p, 'r--', linewidth=2.5, label='拟合正态曲线')
        
        # 使用全局字体配置，无需重复指定路径
        ax.set_title(
            f"{dist_type} 样本容量 n={anim_n} 时的均值收敛演示",
            fontsize=16, fontweight='bold'
        )
        ax.set_xlabel("样本均值数值", fontsize=12)
        ax.set_ylabel("概率密度", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # 先渲染图表，再关闭，避免资源泄漏
        with chart_placeholder:
            st.pyplot(fig)
        plt.close(fig)  # 立即释放图表资源

        # 计算统计指标
        sk = skew(anim_sample_means)
        kurt = kurtosis(anim_sample_means)
        abs_sk = abs(sk)
        skewness_color = "#2ecc71" if abs_sk < 0.5 else "#f1c40f" if 0.5 <= abs_sk <= 1 else "#e74c3c"
        
        # 更新统计指标
        with stats_placeholder:
            st.subheader("📊 实时统计指标（动画演示中）")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("样本均值期望 (Mean)", f"{mu_fit:.4f}")
            with c2:
                st.metric("样本均值标准差 (Std)", f"{std_fit:.4f}")
            with c3:
                st.markdown(f"""
                <div style="background-color: var(--st-card-bg-color); padding: 1rem; border-radius: 0.5rem; height: 100%;">
                    <div style="font-size: 14px; color: var(--st-text-secondary-color); margin-bottom: 0.25rem;">分布偏度 (Skewness)</div>
                    <div style="font-size: 20px; font-weight: 600; color: {skewness_color};">{sk:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.metric("分布峰度 (Kurtosis)", f"{kurt:.4f}")
            with c5:
                normality = "✅ 接近正态" if abs_sk < 0.5 else "❌ 偏离正态"
                st.metric("正态性判断", normality)
        
        # 缩短延时，更流畅
        time.sleep(0.5)
    
    # 动画结束后重置状态
    st.session_state.anim_running = False

# ===================== 底部说明 =====================
st.markdown("---")
st.markdown("""
### 📝 使用说明
1.  左侧可选择不同的母体分布类型，并调节对应参数；
2.  调整样本容量 n 和模拟次数 N，观察均值分布的收敛效果；
3.  偏度越接近0，峰度越接近3，说明分布越对称（越接近正态分布）；
4.  点击「开始动画演示」按钮，可自动观看 n 从1到500的渐进收敛过程，支持中途停止。
""")


