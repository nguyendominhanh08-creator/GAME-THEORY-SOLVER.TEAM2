import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Import core modules
from modules.saddle_point import find_saddle_points
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game

# ---------------------------------------------------------
# 1. Page Configuration & Custom Academic CSS Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="EquiMatrix - Game Theory Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS for Executive Academic Look
st.markdown("""
<style>
    /* Main Background & Title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 500;
    }
    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #1E40AF;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: 800;
    }
    /* Section Divider */
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-color: #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">⚖️ EQUIMATRIX SOLVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Player Zero-Sum Game Theory Optimization Engine • Linear Algebra Project</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-headers/100/matrix.png", width=70)
    st.title("⚙️ Control Panel")
    
    st.markdown("### 📐 Matrix Dimensions")
    rows = st.number_input("Rows $m$ (Player 1 Strategies):", min_value=2, max_value=6, value=2, step=1)
    cols = st.number_input("Cols $n$ (Player 2 Strategies):", min_value=2, max_value=6, value=2, step=1)
    
    st.markdown("---")
    st.markdown("### 📌 Real-World Presets")
    preset = st.selectbox(
        "Choose Test Scenario:",
        [
            "Custom Input Matrix",
            "Scenario 1: Marketing Competition (Saddle Point)",
            "Scenario 2: Contract Bidding (2x2 Mixed)",
            "Scenario 3: Facility Security (3x3 General LP)"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.caption("**Linear Algebra Group Project**")
    st.caption("Version 2.5 • Professional Academic Edition")

# ---------------------------------------------------------
# 4. Input Matrix Handling
# ---------------------------------------------------------
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

col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.subheader("📝 Payoff Matrix Input $A$")
    st.caption("Edit values directly in the grid below:")
    edited_df = st.data_editor(
        default_df,
        key=f"editor_{preset}_{rows}_{cols}",
        num_rows="fixed",
        use_container_width=True
    )

with col_right:
    st.subheader("📊 Matrix Diagnostics")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Matrix Size</div>
        <div class="metric-value">{rows} × {cols}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Total Pure Strategies</div>
        <div class="metric-value">{rows + cols}</div>
    </div>
    """, unsafe_allow_html=True)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 5. Solver & Interactive Results Dashboard
# ---------------------------------------------------------
if solve_btn:
    A = edited_df.to_numpy(dtype=float)
    st.markdown("---")
    st.header("🎯 Equilibrium Results & Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "💡 Strategic Decision Analysis", "📐 Mathematical Proof"])
    
    # Check Saddle Point
    has_sp = False
    saddle_pts = []
    try:
        sp_result = find_saddle_points(A)
        if isinstance(sp_result, dict) and sp_result.get("has_saddle_point"):
            has_sp = True
            saddle_pts = sp_result.get("saddle_points")
        elif isinstance(sp_result, list) and len(sp_result) > 0:
            has_sp = True
            saddle_pts = sp_result
    except Exception:
        pass

    if has_sp:
        r_idx, c_idx = saddle_pts[0]
        game_val = A[r_idx, c_idx]
        
        with tab1:
            st.success("✅ Pure Strategy Equilibrium (Saddle Point) Detected!")
            m1, m2, m3 = st.columns(3)
            m1.metric("Game Value (v)", f"{game_val:.4f}")
            m2.metric("Player 1 Optimal Action", f"Row {r_idx + 1} (X{r_idx + 1})")
            m3.metric("Player 2 Optimal Action", f"Col {c_idx + 1} (Y{c_idx + 1})")
            st.info(f"**Saddle Point Coordinate:** Row {r_idx + 1}, Column {c_idx + 1}")

        with tab2:
            st.markdown(f"### 🎯 Pure Strategy Recommendation")
            st.write(f"- **Player 1** should deterministically select **Strategy X{r_idx+1}** (Probability = 100%).")
            st.write(f"- **Player 2** should deterministically select **Strategy Y{c_idx+1}** (Probability = 100%).")
            st.write(f"- **Expected Payoff ($v$):** {game_val:.4f}")

        with tab3:
            st.latex(r"\max_i \min_j a_{ij} = \min_j \max_i a_{ij} = v")
            st.write(f"The maximin payoff equals the minimax payoff at value **{game_val:.4f}**.")

    else:
        # Mixed Strategy Solutions
        x_opt, y_opt, game_val, solver_name = None, None, None, ""
        
        if rows == 2 and cols == 2:
            try:
                x_opt, y_opt, game_val = solve_game_2x2(A)
                solver_name = "2x2 Analytical Algebraic Module"
            except Exception:
                pass
                
        if x_opt is None:
            sol = solve_matrix_game(A)
            x_opt, y_opt, game_val = sol.x, sol.y, sol.value
            solver_name = "General m × n Linear Programming Engine (HiGHS)"

        with tab1:
            st.warning(f"⚠️ No Saddle Point found. Solved via **{solver_name}**.")
            st.metric("Expected Game Value (v)", f"{game_val:.4f}")
            
            st.markdown("### 📈 Interactive Strategy Distribution")
            chart_col1, chart_col2 = st.columns(2)
            
            # Interactive Plotly Bar Chart for Player 1
            df_x = pd.DataFrame({
                "Strategy": [f"X{i+1}" for i in range(rows)],
                "Probability": x_opt
            })
            fig_x = px.bar(
                df_x, x="Strategy", y="Probability", text_auto=".4f",
                title="Player 1 Strategy Distribution (x*)",
                color="Probability", color_continuous_scale="Blues"
            )
            fig_x.update_layout(yaxis_range=[0, 1.05], showlegend=False)
            chart_col1.plotly_chart(fig_x, use_container_width=True)

            # Interactive Plotly Bar Chart for Player 2
            df_y = pd.DataFrame({
                "Strategy": [f"Y{j+1}" for j in range(cols)],
                "Probability": y_opt
            })
            fig_y = px.bar(
                df_y, x="Strategy", y="Probability", text_auto=".4f",
                title="Player 2 Strategy Distribution (y*)",
                color="Probability", color_continuous_scale="Reds"
            )
            fig_y.update_layout(yaxis_range=[0, 1.05], showlegend=False)
            chart_col2.plotly_chart(fig_y, use_container_width=True)

        with tab2:
            st.markdown("### 💡 Optimal Randomization Strategy")
            st.write(f"- **Player 1 (Row Player)** optimal probability vector $x^*$: `{np.round(x_opt, 4)}`")
            st.write(f"- **Player 2 (Column Player)** optimal probability vector $y^*$: `{np.round(y_opt, 4)}`")
            st.write(f"- Under minimax equilibrium, the expected long-term payoff is **{game_val:.4f}**.")

        with tab3:
            st.markdown("### 📐 Linear Programming Optimization Formulation")
            st.latex(r"\max v \quad \text{s.t.} \quad A^T x \ge v \mathbf{1}, \quad \sum x_i = 1, \quad x_i \ge 0")
            st.latex(r"\min v \quad \text{s.t.} \quad A y \le v \mathbf{1}, \quad \sum y_j = 1, \quad y_j \ge 0")
