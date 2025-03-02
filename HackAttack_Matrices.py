import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit UI: Sidebar for adding vectors
st.sidebar.header("Vector Operations")

# Store vectors dynamically
if "vectors" not in st.session_state:
    st.session_state.vectors = []

# Button to add a new vector
if st.sidebar.button("➕ Add Vector"):
    st.session_state.vectors.append({"x": 0.0, "y": 0.0, "z": 0.0, "color": "#FF0000"})

# Display vector inputs only if vectors exist
if st.session_state.vectors:
    for i, vec in enumerate(st.session_state.vectors):
        with st.sidebar.expander(f"Vector {i+1}", expanded=True):
            col1, col2, col3 = st.columns(3)
            vec["x"] = col1.number_input(f"x{i+1}", value=vec["x"], key=f"x{i}")
            vec["y"] = col2.number_input(f"y{i+1}", value=vec["y"], key=f"y{i}")
            vec["z"] = col3.number_input(f"z{i+1}", value=vec["z"], key=f"z{i}")
            vec["color"] = st.color_picker("Color", vec["color"], key=f"color{i}")

            # Delete button for each vector
            if st.button(f"❌ Delete Vector {i+1}", key=f"delete{i}"):
                st.session_state.vectors.pop(i)
                st.experimental_rerun()

# Initialize figure
fig = go.Figure()

# Add user vectors to 3D plot **only if they exist**
if st.session_state.vectors:
    for i, vec in enumerate(st.session_state.vectors):
        fig.add_trace(go.Scatter3d(
            x=[0, vec["x"]], y=[0, vec["y"]], z=[0, vec["z"]],
            mode="lines+markers",
            marker=dict(size=5, color=vec["color"]),
            line=dict(width=5, color=vec["color"]),
            name=f"Vector {i+1}"  # <-- Setting proper legend names
        ))

# Fix 3D axes scaling
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], title="X-Axis"),
        yaxis=dict(range=[-10, 10], title="Y-Axis"),
        zaxis=dict(range=[-10, 10], title="Z-Axis"),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1)
    ),
    legend_title="Vectors"  # <-- Update legend title
)

# Display plot
st.plotly_chart(fig)

# Perform vector operations **only if at least two vectors exist**
if len(st.session_state.vectors) >= 2:
    v1 = np.array([st.session_state.vectors[0]["x"], st.session_state.vectors[0]["y"], st.session_state.vectors[0]["z"]])
    v2 = np.array([st.session_state.vectors[1]["x"], st.session_state.vectors[1]["y"], st.session_state.vectors[1]["z"]])

    # Vector arithmetic results
    st.subheader("Vector Operations")
    col1, col2 = st.columns(2)
    col1.write(f"**a + b** = {v1 + v2}")
    col2.write(f"**a - b** = {v1 - v2}")
    col1.write(f"**Dot Product (a · b)** = {np.dot(v1, v2)}")
    col2.write(f"**Cross Product (a × b)** = {np.cross(v1, v2)}")
 