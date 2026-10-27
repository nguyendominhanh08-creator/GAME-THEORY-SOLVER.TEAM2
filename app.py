# 1. Check Saddle Point
    has_sp = False
    saddle_pts = []
    
    try:
        sp_res = find_saddle_points(A)
        # Trường hợp sp_res trả về dict
        if isinstance(sp_res, dict):
            if sp_res.get("has_saddle_point") or len(sp_res.get("saddle_points", [])) > 0:
                has_sp = True
                saddle_pts = sp_res.get("saddle_points", [])
        # Trường hợp sp_res trả về list các tuple [(0, 1)]
        elif isinstance(sp_res, (list, tuple)) and len(sp_res) > 0:
            has_sp = True
            saddle_pts = sp_res
    except Exception:
        # Nếu module saddle_point bị lỗi, tự tính thủ công trực tiếp bằng NumPy
        row_mins = np.min(A, axis=1)
        col_maxs = np.max(A, axis=0)
        maximin = np.max(row_mins)
        minimax = np.min(col_maxs)
        if np.isclose(maximin, minimax):
            has_sp = True
            saddle_pts = [(int(i), int(j)) for i in range(A.shape[0]) for j in range(A.shape[1]) if np.isclose(A[i, j], maximin) and np.isclose(row_mins[i], maximin) and np.isclose(col_maxs[j], minimax)]

    if has_sp and len(saddle_pts) > 0:
        sp_item = saddle_pts[0]
        if isinstance(sp_item, (tuple, list)):
            r_idx, c_idx = int(sp_item[0]), int(sp_item[1])
        elif isinstance(sp_item, dict):
            r_idx, c_idx = int(sp_item.get("row", 0)), int(sp_item.get("col", 0))
        else:
            r_idx, c_idx = 0, 1
            
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

        st.stop() # Dừng chương trình tại đây, không cho chạy xuống cảnh báo No Pure Saddle Point!
