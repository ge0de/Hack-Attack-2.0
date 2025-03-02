
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit UI: Sidebar for adding vectors
st.sidebar.header("Matrix & Vector Operations")

# Store vectors dynamically
if "vectors" not in st.session_state:
    st.session_state.vectors = []

# Button to add a new vector
if st.sidebar.button("➕ Add Vector"):
    st.session_state.vectors.append({"x": 0.0, "y": 0.0, "z": 0.0, "color": "#FF0000"})
    
# Display vector inputs in sidebar
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
            try:
                st.experimental_rerun()
            except:
                pass
# Main visualization options
option = st.sidebar.selectbox("Choose Visualization", ["Custom Matrix", "Span of R³"])
show_area_volume = st.sidebar.radio("Show Area/Volume", ("No", "Yes"))

opacity = 0.5  # Default opacity
if show_area_volume == "Yes" or option == "Span of R³":
    opacity = st.sidebar.slider("Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

def apply_matrix(matrix):
    basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Standard basis
    transformed_vectors = np.dot(matrix, basis_vectors.T).T  # Apply transformation

    traces = []
    for vec, color in zip(transformed_vectors, ['red', 'green', 'blue']):
        traces.append(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+markers',
            marker=dict(size=5, color=color),
            line=dict(width=5, color=color)
        ))

    if show_area_volume == "Yes":
        if matrix.shape == (3, 3):  # 3D transformation (parallelepiped)
            x = [0, transformed_vectors[0, 0], transformed_vectors[1, 0], transformed_vectors[2, 0], 0]
            y = [0, transformed_vectors[0, 1], transformed_vectors[1, 1], transformed_vectors[2, 1], 0]
            z = [0, transformed_vectors[0, 2], transformed_vectors[1, 2], transformed_vectors[2, 2], 0]

            faces = [
                [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
            ]

            for face in faces:
                traces.append(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[face[0]], j=[face[1]], k=[face[2]],
                    opacity=opacity, color='rgba(0, 100, 255, 0.5)'  # Uses selected opacity
                ))

    return traces

# Function to display span of R³
def show_span():
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(-2, 2, 10), np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    return go.Scatter3d(
        x=grid_x.flatten(), y=grid_y.flatten(), z=grid_z.flatten(),
        mode='markers', marker=dict(size=3, color='blue', opacity=opacity)  # Uses selected opacity
    )

# Initialize figure
fig = go.Figure()

if option == "Custom Matrix":
    st.sidebar.text("Enter Matrix (3x3)")
    matrix = np.array([
        [st.sidebar.number_input(f"Row {i+1}, Col 1", value=1.0) for i in range(3)],
        [st.sidebar.number_input(f"Row {i+1}, Col 2", value=0.0) for i in range(3)],
        [st.sidebar.number_input(f"Row {i+1}, Col 3", value=0.0) for i in range(3)]
    ])
    for trace in apply_matrix(matrix):
        fig.add_trace(trace)
elif option == "Span of R³":
    fig.add_trace(show_span())

# Add user vectors to 3D plot
for vec in st.session_state.vectors:
    fig.add_trace(go.Scatter3d(
        x=[0, vec["x"]], y=[0, vec["y"]], z=[0, vec["z"]],
        mode="lines+markers",
        marker=dict(size=5, color=vec["color"]),
        line=dict(width=5, color=vec["color"])
    ))

# Fix 3D axes scaling
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-10, 10], title="X-Axis"),
        yaxis=dict(range=[-10, 10], title="Y-Axis"),
        zaxis=dict(range=[-10, 10], title="Z-Axis"),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1)
    )
)

# Display plot
st.plotly_chart(fig)

# Perform vector operations if at least two vectors exist
if len(st.session_state.vectors) >= 2:
    v1 = np.array([st.session_state.vectors[0]["x"], st.session_state.vectors[0]["y"], st.session_state.vectors[0]["z"]])
    v2 = np.array([st.session_state.vectors[1]["x"], st.session_state.vectors[1]["y"], st.session_state.vectors[1]["z"]])

    # Vector arithmetic results
    st.subheader("Vector Operations")
    col1, col2 = st.columns(2)
    col1.write(f"*a + b* = {v1 + v2}")
    col2.write(f"*a - b* = {v1 - v2}")
    col1.write(f"*Dot Product (a · b)* = {np.dot(v1, v2)}")
    col2.write(f"*Cross Product (a × b)* = {np.cross(v1, v2)}")