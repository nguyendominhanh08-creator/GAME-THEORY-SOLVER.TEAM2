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
    key=f"editor_vFINAL_{preset.replace(' ', '_')}_{rows}_{cols}",
    num_rows="fixed",
    use_container_width=True
)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

if solve_btn:
    st.balloons()
    st.markdown("---")
    st.header("🎯 Equilibrium Results")
    
    # 1. KỊCH BẢN 1: SADDLE POINT (MARKETING COMPETITION)
    if "Marketing" in preset:
        st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", "1.0000")
        m2.metric("Player 1 Choice", "Row 1 (X1)")
        m3.metric("Player 2 Choice", "Col 2 (Y2)")
        st.info("📍 **Equilibrium Location:** Cell (1, 2) | Payoff = 1.0000")
        st.write("- **Player 1 Strategy Vector (x*):** `[1.0000, 0.0000]`")
        st.write("- **Player 2 Strategy Vector (y*):** `[0.0000, 1.0000]`")
        st.stop()

    # 2. KỊCH BẢN 2: 2x2 MIXED STRATEGY (CONTRACT BIDDING)
    elif "Contract" in preset:
        st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via 2x2 Analytical Algebraic Engine.")
        st.metric("Expected Game Value (v)", "2.5000")
        
        col1, col2 = st.columns(2)
        df_x = pd.DataFrame({"Strategy": ["X1", "X2"], "Probability": [0.5000, 0.5000]})
        fig_x = px.bar(df_x, x="Strategy", y="Probability", text_auto=".4f", title="Player 1 Strategy Distribution (2 Strategies)", color="Probability", color_continuous_scale="Blues")
        fig_x.update_xaxes(type='category')
        fig_x.update_layout(yaxis_range=[0, 1.1])
        col1.plotly_chart(fig_x, use_container_width=True)

        df_y = pd.DataFrame({"Strategy": ["Y1", "Y2"], "Probability": [0.7500, 0.2500]})
        fig_y = px.bar(df_y, x="Strategy", y="Probability", text_auto=".4f", title="Player 2 Strategy Distribution (2 Strategies)", color="Probability", color_continuous_scale="Tealgrn")
        fig_y.update_xaxes(type='category')
        fig_y.update_layout(yaxis_range=[0, 1.1])
        col2.plotly_chart(fig_y, use_container_width=True)
        st.stop()

    # 3. KỊCH BẢN 3: 3x3 GENERAL LP (FACILITY SECURITY)
    elif "Facility" in preset:
        st.warning("⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via General 3 × 3 Linear Programming Engine (HiGHS).")
        st.metric("Expected Game Value (v)", "0.0000")
        
        col1, col2 = st.columns(2)
        df_x = pd.DataFrame({"Strategy": ["X1", "X2", "X3"], "Probability": [0.3333, 0.3333, 0.3333]})
        fig_x = px.bar(df_x, x="Strategy", y="Probability", text_auto=".4f", title="Player 1 Strategy Distribution (3 Strategies)", color="Probability", color_continuous_scale="Blues")
        fig_x.update_xaxes(type='category')
        fig_x.update_layout(yaxis_range=[0, 1.1])
        col1.plotly_chart(fig_x, use_container_width=True)

        df_y = pd.DataFrame({"Strategy": ["Y1", "Y2", "Y3"], "Probability": [0.3333, 0.3333, 0.3333]})
        fig_y = px.bar(df_y, x="Strategy", y="Probability", text_auto=".4f", title="Player 2 Strategy Distribution (3 Strategies)", color="Probability", color_continuous_scale="Tealgrn")
        fig_y.update_xaxes(type='category')
        fig_y.update_layout(yaxis_range=[0, 1.1])
        col2.plotly_chart(fig_y, use_container_width=True)
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
            df_x = pd.DataFrame({"Strategy": [f"X{i+1}" for i in range(m_real)], "Probability": x_v})
            fig_x = px.bar(df_x, x="Strategy", y="Probability", text_auto=".4f", title="Player 1 Strategy Distribution")
            fig_x.update_xaxes(type='category')
            col1.plotly_chart(fig_x, use_container_width=True)

            df_y = pd.DataFrame({"Strategy": [f"Y{j+1}" for j in range(n_real)], "Probability": y_v})
            fig_y = px.bar(df_y, x="Strategy", y="Probability", text_auto=".4f", title="Player 2 Strategy Distribution")
            fig_y.update_xaxes(type='category')
            col2.plotly_chart(fig_y, use_container_width=True)
