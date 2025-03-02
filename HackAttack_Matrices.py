import streamlit as st  # Import Streamlit for UI components
import plotly.graph_objects as go  # Import Plotly for 3D visualization
import numpy as np  # Import NumPy for matrix operations


# Streamlit UI: Add controls for user interaction
st.sidebar.header("Matrix Visualizations")
option = st.sidebar.selectbox("Choose Visualization", ["Custom Matrix", "Span of R³"])
opacity = st.sidebar.slider("Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
show_area_volume = st.sidebar.radio("Show Area/Volume", ("No", "Yes"))

# Function to visualize a custom transformation matrix
def apply_matrix(matrix):
    basis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Standard basis vectors
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

<<<<<<< Updated upstream
            # Define the faces of the parallelepiped with correct vertex indices
            faces = [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]

            for face in faces:
                traces.append(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[face[0], face[1], face[2]],
                    j=[face[1], face[2], face[3]],
                    k=[face[2], face[3], face[0]],
                    opacity=0.2, color='rgba(0, 100, 255, 0.5)'  # Semi-transparent volume
                ))
=======
# Add Vector Button
if st.sidebar.button("➕ Add Vector"):
    add_vector()
matrix = np.array([
    [st.sidebar.number_input(f"Row {i + 1}, Col {j + 1}", value=(1.0 if i == j else 0.0), format="%.2f") for j in range(3)]
    for i in range(3)
], dtype=float)

# Display and Edit Vectors
for i, vector in enumerate(st.session_state.vectors):
    with st.sidebar.expander(f"Vector {i+1}"):

        # Vector Component Inputs
        vector["coords"] = [
            st.number_input(f"X{i+1}", -10.0, 10.0, float(vector["coords"][0]), 0.1),
            st.number_input(f"Y{i+1}", -10.0, 10.0, float(vector["coords"][1]), 0.1)
        ]
        if dimension == "3D":
            vector["coords"].append(st.number_input(f"Z{i+1}", -10.0, 10.0, vector["coords"][2], 0.1))
>>>>>>> Stashed changes

        elif matrix.shape == (2, 2):  # 2D transformation (parallelogram)
            x = [0, transformed_vectors[0, 0], transformed_vectors[1, 0], 0]
            y = [0, transformed_vectors[0, 1], transformed_vectors[1, 1], 0]

            traces.append(go.Scatter(
                x=x + [x[0]], y=y + [y[0]], mode='lines', fill='toself',
                fillcolor='rgba(0, 100, 255, 0.3)', line=dict(color='blue', width=3)
            ))

    return traces


# Function to display the span of R³
def show_span():
    # Create a grid of points
    grid_x, grid_y, grid_z = np.meshgrid(np.linspace(-2, 2, 10),
                                          np.linspace(-2, 2, 10),
                                          np.linspace(-2, 2, 10))

    # Flatten the grid points to create a list of coordinates
    x_flat = grid_x.flatten()
    y_flat = grid_y.flatten()
    z_flat = grid_z.flatten()

    # Plot each point as a marker
    return go.Scatter3d(
        x=x_flat, y=y_flat, z=z_flat,
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5)
    )

fig = go.Figure()

if option == "Custom Matrix":
    st.sidebar.text("Enter Matrix (3x3)")
    matrix = np.array([
        [st.sidebar.number_input(f"Row {i + 1}, Col 1", value=1.0) for i in range(3)],
        [st.sidebar.number_input(f"Row {i + 1}, Col 2", value=0.0) for i in range(3)],
        [st.sidebar.number_input(f"Row {i + 1}, Col 3", value=0.0) for i in range(3)]
    ])
    for trace in apply_matrix(matrix):
        fig.add_trace(trace)
elif option == "Span of R³":
    fig.add_trace(show_span())

# Update layout
fig.update_layout(scene=dict(
    xaxis=dict(range=[-10, 10]),
    yaxis=dict(range=[-10, 10]),
    zaxis=dict(range=[-10, 10])
))

# Display plot in Streamlit app
st.plotly_chart(fig)

#Hey guys this is Mehdi's changes


fig.update_layout(scene=dict(
    xaxis=dict(range=[-10, 10]),
    yaxis=dict(range=[-10, 10]),
    zaxis=dict(range=[-10, 10])
))