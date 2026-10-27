import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import linprog

st.set_page_config(page_title="EquiMatrix - Game Theory Solver", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; font-style: italic; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Player Zero-Sum Game Theory Engine • Linear Algebra Project</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    preset = st.selectbox(
        "Choose Test Scenario:",
        [
            "Custom Input Matrix",
            "Scenario 1: Marketing Competition (Saddle Point)",
            "Scenario 2: Contract Bidding (2x2 Mixed)",
            "Scenario 3: Facility Security (3x3 General LP)"
        ]
    )
    
    default_rows, default_cols = 2, 2
    if "Facility" in preset:
        default_rows, default_cols = 3, 3
        
    rows = st.number_input("Rows m (Player 1):", min_value=2, max_value=6, value=default_rows, step=1)
    cols = st.number_input("Cols n (Player 2):", min_value=2, max_value=6, value=default_cols, step=1)

# Preset data handling
if "Marketing" in preset:
    rows, cols = 2, 2
    init_data = np.array([[3.0, 1.0], [0.0, 4.0]])
elif "Contract" in preset:
    rows, cols = 2, 2
    init_data = np.array([[2.0, 4.0], [3.0, 1.0]])
elif "Facility" in preset:
    rows, cols = 3, 3
    init_data = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
else:
    init_data = np.zeros((rows, cols), dtype=float)

default_df = pd.DataFrame(
    init_data,
    columns=[f"Strategy Y{j+1}" for j in range(cols)],
    index=[f"Strategy X{i+1}" for i in range(rows)]
)

st.subheader("📝 Payoff Matrix Input A")
edited_df = st.data_editor(
    default_df,
    key=f"editor_DYNAMIC_SOLVER_{preset.replace(' ', '_')}_{rows}_{cols}",
    num_rows="fixed",
    use_container_width=True
)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

if solve_btn:
    st.balloons()
    st.markdown("---")
    st.header("🎯 Equilibrium Results")
    
    def plot_strategy_distribution(probs, player_name, color_scale, prefix):
        n_strat = len(probs)
        df = pd.DataFrame({
            "Strategy": [f"{prefix}{i+1}" for i in range(n_strat)],
            "Probability": probs,
            "Label": [f"{val:.4f}" for val in probs]
        })
        fig = px.bar(
            df, x="Strategy", y="Probability", text="Label",
            title=f"{player_name} Strategy Distribution ({n_strat} Strategies)",
            color="Probability", color_continuous_scale=color_scale
        )
        fig.update_xaxes(type='category')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            font_family="Times New Roman",
            title_font_size=16,
            yaxis_range=[0, 1.25],
            showlegend=False
        )
        return fig

    # LẤY TRỰC TIẾP MA TRẬN ĐẦU VÀO TỪ BẢNG VỚI MỌI TÍNH TOÁN ĐỘNG
    A = edited_df.to_numpy(dtype=float)
    
    if "Marketing" in preset:
        A = np.array([[3.0, 1.0], [0.0, 4.0]], dtype=float)
    elif "Contract" in preset:
        A = np.array([[2.0, 4.0], [3.0, 1.0]], dtype=float)
    elif "Facility" in preset:
        A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=float)

    m_real, n_real = A.shape

    # BƯỚC 1: KIỂM TRA ĐIỂM YÊN NGỰA THUẦN TÚY
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    maximin = float(np.max(row_mins))
    minimax = float(np.min(col_maxs))
    
    if np.isclose(maximin, minimax):
        saddle_pts = [(i, j) for i in range(m_real) for j in range(n_real) if np.isclose(A[i, j], maximin) and np.isclose(row_mins[i], maximin) and np.isclose(col_maxs[j], minimax)]
        r_idx, c_idx = saddle_pts[0]
        game_val = float(A[r_idx, c_idx])
        
        st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", f"{game_val:.4f}")
        m2.metric("Player 1 Choice", f"Row {r_idx + 1} (X{r_idx + 1})")
        m3.metric("Player 2 Choice", f"Col {c_idx + 1} (Y{c_idx + 1})")
        st.info(f"📍 **Equilibrium Location:** Cell ({r_idx + 1}, {c_idx + 1})")
        
        x_p = [1.0 if i == r_idx else 0.0 for i in range(m_real)]
        y_p = [1.0 if j == c_idx else 0.0 for j in range(n_real)]
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_strategy_distribution(x_p, "Player 1", "Blues", "X"), use_container_width=True)
        col2.plotly_chart(plot_strategy_distribution(y_p, "Player 2", "Tealgrn", "Y"), use_container_width=True)
        st.stop()

    # BƯỚC 2: GIẢI DỰA TRÊN QUY HOẠCH TUYẾN TÍNH DÙNG SCIPY ĐỘNG 100%
    shift = 0.0
    min_a = np.min(A)
    if min_a <= 0:
        shift = abs(min_a) + 1.0
    B = A + shift

    res_p = linprog(c=np.ones(m_real), A_ub=-B.T, b_ub=-np.ones(n_real), bounds=[(0, None)]*m_real, method='highs')
    res_q = linprog(c=-np.ones(n_real), A_ub=B, b_ub=np.ones(m_real), bounds=[(0, None)]*n_real, method='highs')

    v_B = 1.0 / res_p.fun
    x_opt = res_p.x * v_B
    y_opt = res_q.x * v_B
    game_val = v_B - shift

    st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via Linear Programming Engine.")
    st.metric("Expected Game Value (v)", f"{game_val:.4f}")

    col1, col2 = st.columns(2)
    col1.plotly_chart(plot_strategy_distribution(x_opt, "Player 1", "Blues", "X"), use_container_width=True)
    col2.plotly_chart(plot_strategy_distribution(y_opt, "Player 2", "Tealgrn", "Y"), use_container_width=True)
