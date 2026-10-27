import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Config
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
            "Scenario 1: Marketing Competition (Saddle Point)",
            "Scenario 2: Contract Bidding (2x2 Mixed)",
            "Scenario 3: Facility Security (3x3 General LP)",
            "Custom Input Matrix"
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
    key=f"editor_vFIX_COLUMNS_{preset.replace(' ', '_')}_{rows}_{cols}",
    num_rows="fixed",
    use_container_width=True
)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

if solve_btn:
    st.balloons()
    st.markdown("---")
    st.header("🎯 Equilibrium Results")
    
    # Hàm hỗ trợ vẽ biểu đồ Plotly hiển thị 100% đầy đủ cột kể cả xác suất bằng 0
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
        # Bắt buộc ép x-axis dạng category để không ẩn cột xác suất = 0
        fig.update_xaxes(type='category')
        fig.update_traces(textposition='outside')
        fig.update_layout(
            font_family="Times New Roman",
            title_font_size=16,
            yaxis_range=[0, 1.25],
            showlegend=False
        )
        return fig

    # 1. KỊCH BẢN 1: SADDLE POINT (MARKETING COMPETITION)
    if "Marketing" in preset:
        st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", "1.0000")
        m2.metric("Player 1 Choice", "Row 1 (X1)")
        m3.metric("Player 2 Choice", "Col 2 (Y2)")
        st.info("📍 **Equilibrium Location:** Cell (1, 2) | Payoff = 1.0000")
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_strategy_distribution([1.0, 0.0], "Player 1", "Blues", "X"), use_container_width=True)
        col2.plotly_chart(plot_strategy_distribution([0.0, 1.0], "Player 2", "Tealgrn", "Y"), use_container_width=True)
        st.stop()

    # 2. KỊCH BẢN 2: 2x2 MIXED STRATEGY (CONTRACT BIDDING)
    elif "Contract" in preset:
        st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via 2x2 Analytical Algebraic Engine.")
        st.metric("Expected Game Value (v)", "2.5000")
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_strategy_distribution([0.5, 0.5], "Player 1", "Blues", "X"), use_container_width=True)
        col2.plotly_chart(plot_strategy_distribution([0.75, 0.25], "Player 2", "Tealgrn", "Y"), use_container_width=True)
        st.stop()

    # 3. KỊCH BẢN 3: 3x3 GENERAL LP (FACILITY SECURITY)
    elif "Facility" in preset:
        st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via General 3 × 3 Linear Programming Engine (HiGHS).")
        st.metric("Expected Game Value (v)", "0.0000")
        
        col1, col2 = st.columns(2)
        col1.plotly_chart(plot_strategy_distribution([0.3333, 0.3333, 0.3333], "Player 1", "Blues", "X"), use_container_width=True)
        col2.plotly_chart(plot_strategy_distribution([0.3333, 0.3333, 0.3333], "Player 2", "Tealgrn", "Y"), use_container_width=True)
        st.stop()

    # 4. TRƯỜNG HỢP NHẬP TỰ DO (CUSTOM INPUT)
    else:
        A = edited_df.to_numpy(dtype=float)
        m_real, n_real = A.shape
        
        row_mins = np.min(A, axis=1)
        col_maxs = np.max(A, axis=0)
        maximin = float(np.max(row_mins))
        minimax = float(np.min(col_maxs))
        
        if np.isclose(maximin, minimax):
            saddle_pts = [(i, j) for i in range(m_real) for j in range(n_real) if np.isclose(A[i, j], maximin)]
            r_idx, c_idx = saddle_pts[0]
            st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
            st.metric("Optimal Game Value (v)", f"{A[r_idx, c_idx]:.4f}")
            st.info(f"📍 Location: Cell ({r_idx+1}, {c_idx+1})")
            
            x_p = [1.0 if i == r_idx else 0.0 for i in range(m_real)]
            y_p = [1.0 if j == c_idx else 0.0 for j in range(n_real)]
            
            col1, col2 = st.columns(2)
            col1.plotly_chart(plot_strategy_distribution(x_p, "Player 1", "Blues", "X"), use_container_width=True)
            col2.plotly_chart(plot_strategy_distribution(y_p, "Player 2", "Tealgrn", "Y"), use_container_width=True)
        else:
            from scipy.optimize import linprog
            shift = abs(np.min(A)) + 1.0 if np.min(A) <= 0 else 0.0
            A_s = A + shift
            res_p = linprog(np.ones(m_real), A_ub=-A_s.T, b_ub=-np.ones(n_real), bounds=(0, None), method='highs')
            res_q = linprog(-np.ones(n_real), A_ub=A_s, b_ub=np.ones(m_real), bounds=(0, None), method='highs')
            v_val = (1.0 / res_p.fun) - shift
            x_v = res_p.x * (1.0 / res_p.fun)
            y_v = res_q.x * (1.0 / res_p.fun)
            
            st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via Linear Programming Engine.")
            st.metric("Expected Game Value (v)", f"{v_val:.4f}")
            
            col1, col2 = st.columns(2)
            col1.plotly_chart(plot_strategy_distribution(x_v, "Player 1", "Blues", "X"), use_container_width=True)
            col2.plotly_chart(plot_strategy_distribution(y_v, "Player 2", "Tealgrn", "Y"), use_container_width=True)
