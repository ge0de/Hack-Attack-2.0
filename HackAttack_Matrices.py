import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Streamlit UI: Sidebar for operations
st.sidebar.header("Matrix & Vector Operations")
st.sidebar.header("To see 3D visualization, add a vector")

# Store vectors dynamically
if "vectors" not in st.session_state:
    st.session_state.vectors = []

# Store named matrices
if "matrices" not in st.session_state:
    st.session_state.matrices = {}

# New matrix creation UI
st.sidebar.header("Create Matrix")
matrix_name = st.sidebar.text_input("Enter matrix name")
rows = st.sidebar.number_input("Rows", min_value=1, value=1, step=1)
cols = st.sidebar.number_input("Columns", min_value=1, value=1, step=1)

if st.sidebar.button("Create Matrix"):
    if matrix_name and matrix_name not in st.session_state.matrices:
        st.session_state.matrices[matrix_name] = np.zeros((rows, cols))
    elif not matrix_name:
        st.sidebar.error("Please enter a matrix name")
    else:
        st.sidebar.error("Matrix name already exists!")

# Display and edit matrices
for name in list(st.session_state.matrices.keys()):
    matrix = st.session_state.matrices[name]
    rows_mat, cols_mat = matrix.shape
    
    with st.sidebar.expander(f"Matrix {name}", expanded=True):
        # Create dynamic grid for matrix input
        for i in range(rows_mat):
            cols_grid = st.columns(cols_mat)
            for j in range(cols_mat):
                new_value = cols_grid[j].number_input(
                    f"{name}[{i+1},{j+1}]",
                    value=float(matrix[i, j]),
                    key=f"{name}_cell_{i}_{j}"
                )
                matrix[i, j] = new_value
        
        # Update matrix in session state
        st.session_state.matrices[name] = matrix
        
        # Delete button
        if st.button(f"❌ Delete {name}", key=f"delete_{name}"):
            del st.session_state.matrices[name]
            try:

                st.experimental_rerun()
            except:
                pass

# Vector creation and management remains unchanged
if st.sidebar.button("➕ Add Vector"):
    st.session_state.vectors.append({"x": 0.0, "y": 0.0, "z": 0.0, "color": "#FF0000"})

for i, vec in enumerate(st.session_state.vectors):
    with st.sidebar.expander(f"Vector {i+1}", expanded=True):
        col1, col2, col3 = st.columns(3)
        vec["x"] = col1.number_input(f"x{i+1}", value=float(vec["x"]), key=f"x{i}")
        vec["y"] = col2.number_input(f"y{i+1}", value=float(vec["y"]), key=f"y{i}")
        vec["z"] = col3.number_input(f"z{i+1}", value=float(vec["z"]), key=f"z{i}")
        vec["color"] = st.color_picker("Color", vec["color"], key=f"color{i}")

        if st.button(f"❌ Delete Vector {i+1}", key=f"delete{i}"):
            st.session_state.vectors.pop(i)
            try:
                st.experimental_rerun()
            except:
                pass

# Visualization options remain unchanged
option = st.sidebar.selectbox("Choose Visualization", ["None", "Span of R³"])
opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.5, 0.1)
show_area_volume = st.sidebar.radio("Show Area/Volume", ("No", "Yes"))

# 3D visualization code remains unchanged
fig = go.Figure()

# Unit vectors and vector visualization remains unchanged
unit_vectors = [
    ([0, 0], [0, 0], [0, 0], "#FF0000"),
    ([0, 0], [0, 0], [0, 0], "#00FF00"),
    ([0, 0], [0, 0], [0, 0], "#0000FF")
]

for ux, uy, uz, color in unit_vectors:
    fig.add_trace(go.Scatter3d(
        x=ux, y=uy, z=uz,
        mode="lines+markers",
        marker=dict(size=5, color=color),
        line=dict(width=5, color=color)
    ))

vector_points = []
for vec in st.session_state.vectors:
    fig.add_trace(go.Scatter3d(
        x=[0, vec["x"]], y=[0, vec["y"]], z=[0, vec["z"]],
        mode="lines+markers",
        marker=dict(size=5, color=vec["color"]),
        line=dict(width=5, color=vec["color"])
    ))
    vector_points.append([vec["x"], vec["y"], vec["z"]])

# Area/volume calculations remain unchanged
if show_area_volume == "Yes" and len(vector_points) >= 2:
    vector_points = np.array(vector_points)
    if len(vector_points) == 2:
        v1, v2 = vector_points[:2]
        area = np.linalg.norm(np.cross(v1, v2))
        st.write(f"Area of parallelogram: {area}")
    elif len(vector_points) >= 3:
        v1, v2, v3 = vector_points[:3]
        volume = np.abs(np.dot(v1, np.cross(v2, v3)))
        st.write(f"Volume of parallelepiped: {volume}")

# Determinant calculation remains unchanged
if len(st.session_state.vectors) == 3:
    matrix = np.array([[v["x"], v["y"], v["z"]] for v in st.session_state.vectors])
    determinant = np.linalg.det(matrix)
    st.subheader("Determinant of 3x3 Matrix")
    st.write(f"Det(A) = {determinant}")

# Axis configuration remains unchanged
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-2, 2], title="X-Axis"),
        yaxis=dict(range=[-2, 2], title="Y-Axis"),
        zaxis=dict(range=[-2, 2], title="Z-Axis"),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1)
    )
)

st.plotly_chart(fig)

# Matrix operations input remains unchanged
expr = st.text_input("Enter operation (e.g., a + b, a @ b, cross(a, b))")

if expr:
    try:
        safe_dict = {k: v for k, v in st.session_state.matrices.items()}
        safe_dict["cross"] = np.cross
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        st.write(f"Result of `{expr}`:")
        st.write(result)
    except Exception as e:
        st.error(f"Invalid expression: {e}")