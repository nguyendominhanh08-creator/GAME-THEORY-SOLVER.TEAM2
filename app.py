import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Import core modules
from modules.saddle_point import find_saddle_points
from modules.solve_2x2 import solve_game_2x2
from modules.lp_solver import solve_matrix_game

# ---------------------------------------------------------
# 1. Page Configuration & Custom Academic CSS
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
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
        animation: fadeInDown 1s ease-in-out;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
        font-style: italic;
        animation: fadeInUp 1.2s ease-in-out;
    }

    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border-left: 5px solid #1E3A8A;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.15);
    }
    .metric-title {
        font-size: 1rem;
        color: #1E40AF;
        font-weight: bold;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 2rem;
        color: #0F172A;
        font-weight: 800;
    }

    div.stButton > button {
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        transition: all 0.3s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(37, 99, 235, 0.5);
    }

    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Hero Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">⚖️ EQUIMATRIX OPTIMIZATION SOLVER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Two-Player Zero-Sum Game Theory Engine • Linear Algebra Group Project</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Sidebar Setup
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    
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
    
    st.markdown("### 📐 Matrix Dimensions")
    default_rows = 2
    default_cols = 2
    if "Facility" in preset:
        default_rows, default_cols = 3, 3

    rows = st.number_input("Rows $m$ (Player 1 Strategies):", min_value=2, max_value=6, value=default_rows, step=1)
    cols = st.number_input("Cols $n$ (Player 2 Strategies):", min_value=2, max_value=6, value=default_cols, step=1)
    
    st.markdown("---")
    st.markdown("### 🎓 Academic Information")
    st.caption("**Course:** Linear Algebra")
    st.caption("**Version:** 5.1 Categorical Axis Fix")

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
    st.caption("Double-click cells below to edit numerical values:")
    edited_df = st.data_editor(
        default_df,
        key=f"editor_{preset}_{rows}_{cols}",
        num_rows="fixed",
        use_container_width=True
    )

with col_right:
    st.subheader("📊 Matrix Properties")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Matrix Size</div>
        <div class="metric-value">{rows} × {cols}</div>
    </div>
    <div class="metric-card">
        <div class="metric-title">Total Strategy Space</div>
        <div class="metric-value">{rows + cols} Choices</div>
    </div>
    """, unsafe_allow_html=True)

solve_btn = st.button("🚀 Calculate Equilibrium Solution", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 5. Solver & Animated Interactive Results Dashboard
# ---------------------------------------------------------
if solve_btn:
    st.balloons()
    
    A = edited_df.to_numpy(dtype=float)
    m_real, n_real = A.shape
    
    st.markdown("---")
    st.header("🎯 Equilibrium Results & Decision Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Interactive Dashboard", "💡 Strategic Decision Analysis", "📐 Mathematical Formulation"])
    
    # 1. Safe Saddle Point Check
    has_sp = False
    saddle_pts = []
    
    try:
        sp_res = find_saddle_points(A)
        if isinstance(sp_res, dict):
            has_sp = sp_res.get("has_saddle_point", False)
            saddle_pts = sp_res.get("saddle_points", [])
        elif isinstance(sp_res, list):
            has_sp = len(sp_res) > 0
            saddle_pts = sp_res
    except Exception:
        has_sp = False
        saddle_pts = []

    if has_sp and len(saddle_pts) > 0:
        sp_item = saddle_pts[0]
        if isinstance(sp_item, (tuple, list)):
            r_idx, c_idx = int(sp_item[0]), int(sp_item[1])
        elif isinstance(sp_item, dict):
            r_idx, c_idx = int(sp_item.get("row", 0)), int(sp_item.get("col", 0))
        else:
            r_idx, c_idx = 0, 0
            
        game_val = float(A[r_idx, c_idx])
        
        with tab1:
            st.success("✅ Pure Strategy Equilibrium (Saddle Point) Identified!")
            m1, m2, m3 = st.columns(3)
            m1.metric("Optimal Game Value (v)", f"{game_val:.4f}")
            m2.metric("Player 1 Choice", f"Row {r_idx + 1} (X{r_idx + 1})")
            m3.metric("Player 2 Choice", f"Col {c_idx + 1} (Y{c_idx + 1})")
            st.info(f"📍 **Equilibrium Point Location:** Cell ({r_idx + 1}, {c_idx + 1})")

        with tab2:
            st.markdown(f"### 🎯 Optimal Pure Decision")
            st.write(f"- **Player 1** must play **Strategy X{r_idx+1}** with 100% probability.")
            st.write(f"- **Player 2** must play **Strategy Y{c_idx+1}** with 100% probability.")

        with tab3:
            st.latex(r"\max_i \min_j a_{ij} = \min_j \max_i a_{ij} = v")
            st.write(f"The Maximin value equals Minimax value at **{game_val:.4f}**.")

    else:
        # 2. Mixed Strategy Solutions
        x_opt, y_opt, game_val, solver_name = None, None, None, ""
        
        if m_real == 2 and n_real == 2:
            try:
                res_2x2 = solve_game_2x2(A)
                if isinstance(res_2x2, tuple):
                    x_opt, y_opt, game_val = res_2x2
                elif isinstance(res_2x2, dict):
                    x_opt = res_2x2.get("x")
                    y_opt = res_2x2.get("y")
                    game_val = res_2x2.get("value")
                solver_name = "2x2 Analytical Algebraic Module"
            except Exception:
                pass
                
        if x_opt is None:
            try:
                sol = solve_matrix_game(A)
                if hasattr(sol, 'x'):
                    x_opt, y_opt, game_val = sol.x, sol.y, sol.value
                elif isinstance(sol, dict):
                    x_opt = sol.get("x")
                    y_opt = sol.get("y")
                    game_val = sol.get("value")
                else:
                    x_opt, y_opt, game_val = sol[0], sol[1], sol[2]
                solver_name = f"General {m_real} × {n_real} Linear Programming Engine (HiGHS)"
            except Exception as e:
                st.error(f"❌ LP Solver Error: {e}")
                st.stop()

        x_opt = np.asarray(x_opt, dtype=float).flatten()[:m_real]
        y_opt = np.asarray(y_opt, dtype=float).flatten()[:n_real]
        game_val = float(np.squeeze(game_val))

        with tab1:
            st.warning(f"⚠️ No Pure Saddle Point found. Optimal Mixed Equilibrium calculated via **{solver_name}**.")
            st.metric("Expected Game Value (v)", f"{game_val:.4f}")
            
            st.markdown("### 📈 Interactive Strategy Probability Distributions")
            chart_col1, chart_col2 = st.columns(2)
            
            # Plotly Chart Player 1 (Bắt buộc trục X dạng Category)
            df_x = pd.DataFrame({
                "Strategy": [f"X{i+1}" for i in range(m_real)],
                "Probability": x_opt,
                "TextLabel": [f"{val:.4f}" for val in x_opt]
            })
            fig_x = px.bar(
                df_x, x="Strategy", y="Probability", text="TextLabel",
                title=f"Player 1 Strategy Distribution ({m_real} Strategies)",
                color="Probability", color_continuous_scale="Blues"
            )
            fig_x.update_xaxes(type='category') # Bắt buộc hiện đủ nhãn X1, X2...
            fig_x.update_traces(textposition='outside') # Hiện chữ số rõ ràng trên đầu cột
            fig_x.update_layout(
                font_family="Times New Roman",
                title_font_size=18,
                yaxis_range=[0, 1.15],
                showlegend=False
            )
            chart_col1.plotly_chart(fig_x, use_container_width=True)

            # Plotly Chart Player 2 (Bắt buộc trục X dạng Category)
            df_y = pd.DataFrame({
                "Strategy": [f"Y{j+1}" for j in range(n_real)],
                "Probability": y_opt,
                "TextLabel": [f"{val:.4f}" for val in y_opt]
            })
            fig_y = px.bar(
                df_y, x="Strategy", y="Probability", text="TextLabel",
                title=f"Player 2 Strategy Distribution ({n_real} Strategies)",
                color="Probability", color_continuous_scale="Tealgrn"
            )
            fig_y.update_xaxes(type='category') # Bắt buộc hiện đủ nhãn Y1, Y2...
            fig_y.update_traces(textposition='outside') # Hiện chữ số rõ ràng trên đầu cột
            fig_y.update_layout(
                font_family="Times New Roman",
                title_font_size=18,
                yaxis_range=[0, 1.15],
                showlegend=False
            )
            chart_col2.plotly_chart(fig_y, use_container_width=True)

        with tab2:
            st.markdown("### 💡 Strategic Takeaways")
            st.write(f"- **Player 1 Strategy Vector ($x^*$):** `{np.round(x_opt, 4)}`")
            st.write(f"- **Player 2 Strategy Vector ($y^*$):** `{np.round(y_opt, 4)}`")
            st.write(f"- Expected game payoff under optimal mixed strategy equilibrium is **{game_val:.4f}**.")

        with tab3:
            st.markdown("### 📐 Linear Programming Model Formulation")
            st.latex(r"\max v \quad \text{subject to} \quad A^T x \ge v \mathbf{1}, \quad \sum x_i = 1, \quad x_i \ge 0")
            st.latex(r"\min v \quad \text{subject to} \quad A y \le v \mathbf{1}, \quad \sum y_j = 1, \quad y_j \ge 0")
