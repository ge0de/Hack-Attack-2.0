import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Vector Visualizer", layout="wide")

# Initialize session state if it doesn't exist
if "vectors" not in st.session_state:
    st.session_state.vectors = []

# Sidebar Controls
st.sidebar.header("Vector Controls")
dimension = st.sidebar.radio("Choose Dimension", ["2D", "3D"])
opacity = st.sidebar.slider("Vector Opacity", 0.1, 1.0, 0.8, 0.1)

# Function to add a new vector
def add_vector():
    default_vector = [1, 1, 1] if dimension == "3D" else [1, 1]
    st.session_state.vectors.append({"coords": default_vector, "color": "#FF0000"})

# Function to remove a vector
def remove_vector(index):
    del st.session_state.vectors[index]

# Add Vector Button
if st.sidebar.button("➕ Add Vector"):
    add_vector()

# Display and Edit Vectors
for i, vector in enumerate(st.session_state.vectors):
    with st.sidebar.expander(f"Vector {i+1}"):
        # Vector Component Inputs
        vector["coords"] = [
            st.slider(f"X(i+1)", -10.0, 10.0, vector["coords"][0], 0.1),
            st.slider(f"Y{i+1}", -10.0, 10.0, vector["coords"][1], 0.1)
        ]
        if dimension == "3D":
            vector["coords"].append(st.slider(f"Z{i+1}", -10.0, 10.0, vector["coords"][2], 0.1))

        # Color Picker
        vector["color"] = st.color_picker(f"Color {i+1}", vector["color"])

        # Delete Vector Button
        if st.button(f"❌ Remove Vector {i+1}", key=f"remove_{i}"):
            remove_vector(i)
            st.rerun()

# Visualization
fig = go.Figure()

# Add vectors to plot
for vector in st.session_state.vectors:
    coords = vector["coords"]
    color = vector["color"]
    
    fig.add_trace(go.Scatter3d(
        x=[0, coords[0]], y=[0, coords[1]], z=[0, coords[2]] if dimension == "3D" else [0, 0],
        mode="lines+markers",
        marker=dict(size=6, color=color),
        line=dict(width=5, color=color),
        opacity=opacity
    ))

# Set 2D or 3D layout
if dimension == "2D":
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(visible=False)
        )
    )
else:
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-10, 10])
        )
    )

st.plotly_chart(fig)
