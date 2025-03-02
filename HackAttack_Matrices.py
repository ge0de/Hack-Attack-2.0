import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit UI: Sidebar for adding vectors and matrices
st.sidebar.header("Matrix & Vector Operations")
st.sidebar.header("To see 3D visualization, add a vector")

# Store vectors dynamically
if "vectors" not in st.session_state:
    st.session_state.vectors = []

# Store named matrices/vectors
if "matrices" not in st.session_state:
    st.session_state.matrices = {}

# Function to add a new matrix/vector
def add_matrix(name):
    if name and name not in st.session_state.matrices:
        st.session_state.matrices[name] = np.eye(3)  # Default: Identity matrix

# Input for naming matrices/vectors
matrix_name = st.sidebar.text_input("Enter name (e.g., a, b, c)")

# Button to create matrix
if st.sidebar.button("➕ Add Matrix/Vector"):
    add_matrix(matrix_name)

# Display matrices in the sidebar
for name, matrix in list(st.session_state.matrices.items()):
    with st.sidebar.expander(f"Matrix/Vector {name}", expanded=True):
        updated_matrix = np.copy(matrix)  # Create a copy to update values
        cols = [st.columns(3) for _ in range(3)]  # Create 3x3 grid
        for i in range(3):
            for j in range(3):
                updated_matrix[i, j] = cols[i][j].number_input(f"{name}[{i+1},{j+1}]", value=float(matrix[i, j]), key=f"{name}{i}{j}")
        st.session_state.matrices[name] = updated_matrix  # Save updates
        if st.button(f"❌ Delete {name}", key=f"delete_{name}"):
            del st.session_state.matrices[name]
            st.experimental_rerun()

# Button to add a new vector
if st.sidebar.button("➕ Add Vector"):
    st.session_state.vectors.append({"x": 0.0, "y": 0.0, "z": 0.0, "color": "#FF0000"})

# Display vector inputs in sidebar
for i, vec in enumerate(st.session_state.vectors):
    with st.sidebar.expander(f"Vector {i+1}", expanded=True):
        col1, col2, col3 = st.columns(3)
        vec["x"] = col1.number_input(f"x{i+1}", value=float(vec["x"]), key=f"x{i}")
        vec["y"] = col2.number_input(f"y{i+1}", value=float(vec["y"]), key=f"y{i}")
        vec["z"] = col3.number_input(f"z{i+1}", value=float(vec["z"]), key=f"z{i}")
        vec["color"] = st.color_picker("Color", vec["color"], key=f"color{i}")

        # Delete button for each vector
        if st.button(f"❌ Delete Vector {i+1}", key=f"delete{i}"):
            st.session_state.vectors.pop(i)
            st.experimental_rerun()

# Main visualization options
option = st.sidebar.selectbox("Choose Visualization", ["None", "Span of R³"])  # Added "None" as default
opacity = st.sidebar.slider("Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
show_area_volume = st.sidebar.radio("Show Area/Volume", ("No", "Yes"))

# Initialize figure
fig = go.Figure()

# Add unit vectors
unit_vectors = [
    ([0, 1], [0, 0], [0, 0], "#FF0000"),  # X-axis (Red)
    ([0, 0], [0, 1], [0, 0], "#00FF00"),  # Y-axis (Green)
    ([0, 0], [0, 0], [0, 1], "#0000FF")   # Z-axis (Blue)
]
for ux, uy, uz, color in unit_vectors:
    fig.add_trace(go.Scatter3d(
        x=ux, y=uy, z=uz,
        mode="lines+markers",
        marker=dict(size=5, color=color),
        line=dict(width=5, color=color)
    ))

# Add user vectors to 3D plot
vector_points = []
for vec in st.session_state.vectors:
    fig.add_trace(go.Scatter3d(
        x=[0, vec["x"]], y=[0, vec["y"]], z=[0, vec["z"]],
        mode="lines+markers",
        marker=dict(size=5, color=vec["color"]),
        line=dict(width=5, color=vec["color"])
    ))
    vector_points.append([vec["x"], vec["y"], vec["z"]])

# Show parallelogram or parallelepiped if Show Area/Volume is enabled
if show_area_volume == "Yes" and len(vector_points) >= 2:
    vector_points = np.array(vector_points)
    if len(vector_points) == 2:  # 2D case
        v1, v2 = vector_points[:2]
        area = np.linalg.norm(np.cross(v1, v2))
        st.write(f"Area of parallelogram: {area}")
    elif len(vector_points) >= 3:  # 3D case
        v1, v2, v3 = vector_points[:3]
        volume = np.abs(np.dot(v1, np.cross(v2, v3)))
        st.write(f"Volume of parallelepiped: {volume}")

# Calculate determinant if exactly three vectors exist
if len(st.session_state.vectors) == 3:
    matrix = np.array([[v["x"], v["y"], v["z"]] for v in st.session_state.vectors])
    determinant = np.linalg.det(matrix)
    st.subheader("Determinant of 3x3 Matrix")
    st.write(f"Det(A) = {determinant}")

# Fix 3D axes scaling
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-2, 2], title="X-Axis"),
        yaxis=dict(range=[-2, 2], title="Y-Axis"),
        zaxis=dict(range=[-2, 2], title="Z-Axis"),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1)
    )
)

# Display plot
st.plotly_chart(fig)

# Expression input for matrix operations
expr = st.text_input("Enter operation (e.g., a + b, a @ b, cross(a, b))")

# Evaluate and display result
if expr:
    try:
        safe_dict = {k: v for k, v in st.session_state.matrices.items()}
        safe_dict["cross"] = np.cross  # Allow cross product
        result = eval(expr, {"__builtins__": {}}, safe_dict)  # Safe eval
        st.write(f"Result of `{expr}`:")
        st.write(result)
    except Exception as e:
        st.error(f"Invalid expression: {e}")
