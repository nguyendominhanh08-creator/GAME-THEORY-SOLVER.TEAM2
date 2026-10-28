import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from modules.saddle_point import find_saddle_points
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game

# ---------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, button, input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1E3A8A; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #4B5563; text-align: center; font-style: italic; margin-bottom: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚖️ EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Player Zero-Sum Game Theory Engine • Linear Algebra Project</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Sidebar Setup & Scenario Presets
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    preset = st.selectbox(
        "Choose Test Scenario:",
        [
            "Custom Square Matrix Input",
            "Scenario 1: Marketing Competition (2x2 Saddle Point)",
            "Scenario 2: Contract Bidding (2x2 Mixed Strategy)",
            "Scenario 3: Facility Security (3x3 General LP)"
        ]
    )
    
    # Ép ma trận về dạng vuông N x N để an toàn tuyệt đối
    if "Facility" in preset:
        dim = 3
    elif "Scenario 1" in preset or "Scenario 2" in preset:
        dim = 2
    else:
        dim = st.number_input("Square Matrix Dimension (N x N):", min_value=2, max_value=6, value=3, step=1)

# Preset Initial Data
if "Scenario 1" in preset:
    dim = 2
    init_data = np.array([[3.0, 1.0], [0.0, 4.0]])
elif "Scenario 2" in preset:
    dim = 2
    init_data = np.array([[2.0, 4.0], [3.0, 1.0]])
elif "Scenario 3" in preset:
    dim = 3
    init_data = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
else:
    init_data = np.zeros((dim, dim), dtype=float)

default_df = pd.DataFrame(
    init_data,
    columns=[f"Strategy Y{j+1}" for j in range(dim)],
    index=[f"Strategy X{i+1}" for i in range(dim)]
)

st.subheader("📝 Payoff Matrix Input A")
edited_df = st.data_editor(
    default_df,
    key=f"editor_STRICT_{preset}_{dim}",
    num_rows="fixed",
    use_container_width=True
)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. Plotly Chart Helper Function
# ---------------------------------------------------------
def plot_strategy_distribution(probs, player_name, color_scale, labels):
    probs = np.asarray(probs, dtype=float)
    labels = list(labels)

    if len(probs) != len(labels):
        raise ValueError(f"{player_name}: probability count mismatch.")

    df = pd.DataFrame({
        "Strategy": labels,
        "Probability": probs,
        "Label": [f"{value:.4f}" for value in probs],
    })

    fig = px.bar(
        df,
        x="Strategy",
        y="Probability",
        text="Label",
        title=f"{player_name} Strategy Distribution",
        color="Probability",
        color_continuous_scale=color_scale,
        category_orders={"Strategy": labels},
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        font_family="Times New Roman",
        title_font_size=16,
        showlegend=False,
        yaxis=dict(
            range=[0, 1.25],
            title="Probability",
            tickformat=".2f",
        ),
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=labels,
        ),
        coloraxis_showscale=False,
        margin=dict(t=65, b=50),
    )

    return fig

# ---------------------------------------------------------
# 4. Solver Execution
# ---------------------------------------------------------
if solve_btn:
    st.balloons()
    st.markdown("---")
    st.header("🎯 Equilibrium Results")
    
    A = edited_df.to_numpy(dtype=float)
    n_size = A.shape[0]
    
    # 1. Saddle Point Check
    sp_result = find_saddle_points(A)
    has_sp = False
    saddle_pts = []
    
    if isinstance(sp_result, dict) and sp_result.get("has_saddle_point"):
        has_sp = True
        saddle_pts = sp_result.get("saddle_points")
    elif isinstance(sp_result, list) and len(sp_result) > 0:
        has_sp = True
        saddle_pts = sp_result

    if has_sp:
        r_idx, c_idx = saddle_pts[0]
        game_val = float(A[r_idx, c_idx])
        
        st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Optimal Game Value (v)", f"{game_val:.4f}")
        m2.metric("Player 1 Choice", f"Strategy X{r_idx + 1}")
        m3.metric("Player 2 Choice", f"Strategy Y{c_idx + 1}")
        st.info(f"📍 **Equilibrium Location:** Cell ({r_idx + 1}, {c_idx + 1}) | Payoff = {game_val:.4f}")
        
        x_p = [1.0 if i == r_idx else 0.0 for i in range(n_size)]
        y_p = [1.0 if j == c_idx else 0.0 for j in range(n_size)]
        
        col1, col2 = st.columns(2)
        fig_x = plot_strategy_distribution(x_p, "Player 1", "Blues", [f"Strategy X{i+1}" for i in range(n_size)])
        fig_y = plot_strategy_distribution(y_p, "Player 2", "Tealgrn", [f"Strategy Y{j+1}" for j in range(n_size)])
        col1.plotly_chart(fig_x, use_container_width=True)
        col2.plotly_chart(fig_y, use_container_width=True)

    else:
        # Mixed Strategy Solver
        x_opt, y_opt, game_val, solver_used = None, None, None, ""
        
        # 2. Try 2x2 Analytical Module
        if n_size == 2:
            try:
                x_opt, y_opt, game_val = solve_game_2x2(A)
                solver_used = "2x2 Mixed Strategy Analytical Engine"
            except Exception:
                pass
                
        # 3. General LP Solver Fallback
        if x_opt is None:
            sol = solve_matrix_game(A)
            x_opt, y_opt, game_val = sol.x, sol.y, sol.value
            solver_used = f"General {n_size} × {n_size} Linear Programming Engine (HiGHS)"

        st.warning(f"⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via {solver_used}.")
        st.metric("Expected Game Value (v)", f"{game_val:.4f}")

        labels_x = [f"Strategy X{i+1}" for i in range(n_size)]
        labels_y = [f"Strategy Y{j+1}" for j in range(n_size)]

        col1, col2 = st.columns(2)
        fig_x = plot_strategy_distribution(x_opt, "Player 1", "Blues", labels_x)
        fig_y = plot_strategy_distribution(y_opt, "Player 2", "Tealgrn", labels_y)
        col1.plotly_chart(fig_x, use_container_width=True)
        col2.plotly_chart(fig_y, use_container_width=True)
