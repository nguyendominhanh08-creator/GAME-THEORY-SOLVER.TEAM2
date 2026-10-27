import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from modules.saddle_point import find_saddle_points
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game

st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Times New Roman
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
    key=f"editor_v9_{preset}_{rows}_{cols}",
    num_rows="fixed",
    use_container_width=True
)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

if solve_btn:
    st.balloons()
    
    # Nạp đúng ma trận theo Preset
    if "Marketing" in preset:
        A = np.array([[3.0, 1.0], [0.0, 4.0]], dtype=float)
    elif "Contract" in preset:
        A = np.array([[2.0, 4.0], [3.0, 1.0]], dtype=float)
    elif "Facility" in preset:
        A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]], dtype=float)
    else:
        A = edited_df.to_numpy(dtype=float)
        
    m_real, n_real = A.shape
    st.markdown("---")
    st.header("🎯 Equilibrium Results")
    
    # KÍCH HOẠT HÀM SADDLE POINT
    saddle_pts = find_saddle_points(A)
    
    # NẾU CÓ ĐIỂM YÊN NGỰA
    if len(saddle_pts) > 0:
        r_idx, c_idx = saddle_pts[0]
        game_val = float(A[r_idx, c_idx])
        
        st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", f"{game_val:.4f}")
        m2.metric("Player 1 Choice", f"Row {r_idx + 1} (X{r_idx + 1})")
        m3.metric("Player 2 Choice", f"Col {c_idx + 1} (Y{c_idx + 1})")
        st.info(f"📍 **Equilibrium Location:** Cell ({r_idx + 1}, {c_idx + 1})")
        
        # NGẮT LUỒNG DỨT ĐIỂM TẠI ĐÂY
        st.stop()

    # NẾU KHÔNG CÓ ĐIỂM YÊN NGỰA MỚI CHẠY XUỐNG ĐÂY
    x_opt, y_opt, game_val = None, None, None
    if m_real == 2 and n_real == 2:
        try:
            x_opt, y_opt, game_val = solve_game_2x2(A)
        except Exception:
            pass
            
    if x_opt is None:
        sol = solve_matrix_game(A)
        x_opt, y_opt, game_val = sol.x, sol.y, sol.value

    st.warning("⚠️ No Pure Saddle Point found. Calculating Mixed Strategy Equilibrium...")
    st.metric("Expected Game Value (v)", f"{game_val:.4f}")
    
    col1, col2 = st.columns(2)
    df_x = pd.DataFrame({"Strategy": [f"X{i+1}" for i in range(m_real)], "Probability": x_opt})
    fig_x = px.bar(df_x, x="Strategy", y="Probability", text_auto=".4f", title="Player 1 Strategy Distribution")
    fig_x.update_xaxes(type='category')
    col1.plotly_chart(fig_x, use_container_width=True)

    df_y = pd.DataFrame({"Strategy": [f"Y{j+1}" for j in range(n_real)], "Probability": y_opt})
    fig_y = px.bar(df_y, x="Strategy", y="Probability", text_auto=".4f", title="Player 2 Strategy Distribution")
    fig_y.update_xaxes(type='category')
    col2.plotly_chart(fig_y, use_container_width=True)
