import math
import numpy as np
import matplotlib.pyplot as plt
import heapq

from itertools import combinations, product
import plotly.graph_objects as go


class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y, z)
        self.parent = parent

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current to end)
        self.f = 0  # Total cost: g + h

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


def find_all_paths(connections, cube_dimensions, min_separation, resolution):
    """Find all collision-free paths for the given connections."""
    width, height, depth = cube_dimensions

    # Define cube bounds
    cube_bounds = (-width/2, width/2, -width/2, width/2, -height/2, height/2)

    paths = []
    failed_paths = []

    for i, (start, end) in enumerate(connections):
        print(f"Finding path {i+1}/{len(connections)}: {start} to {end}")

        # Find path using A*
        path = astar(start, end, cube_bounds, paths, min_separation, resolution)

        if path:
            paths.append(path)
            print(f"  Path found with {len(path)} points")
        else:
            failed_paths.append((start, end))
            print(f"  No valid path found")
    return paths, failed_paths


def astar(start, end, cube_bounds, existing_paths, min_separation, resolution):
    """A* algorithm for 3D path finding with collision avoidance."""
    # Create start and end nodes
    start_node = Node(start)
    end_node = Node(end)

    # Initialize open and closed lists
    open_list = []
    closed_set = set()

    # Add the start node to the open list
    heapq.heappush(open_list, start_node)

    # Create a voxel grid to track occupied space
    # This is an optimization to avoid checking all existing paths at each step
    voxel_grid = create_voxel_grid(existing_paths, min_separation, cube_bounds, resolution)

    # Loop until the end is found or open list is empty
    while open_list:
        # Get the node with the lowest F score
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        # If we reached the end node, reconstruct the path
        if octile_distance(current_node.position, end_node.position) < resolution:
            path = []
            temp = current_node
            while temp is not None:
                path.append(temp.position)
                temp = temp.parent
            return path[::-1]  # Return reversed path (start to end)

        # Generate neighbors
        for neighbor in get_neighbors(current_node, resolution, cube_bounds):
            # Skip if already evaluated or in occupied space
            if (neighbor.position in closed_set or
                is_position_occupied(neighbor.position, voxel_grid, resolution)):
                continue

            # Calculate g, h, and f values
            neighbor.g = current_node.g + octile_distance(current_node.position, neighbor.position)

            # Heuristic: straight-line distance to end
            neighbor.h = octile_distance(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h

            # Check if this node is already in the open list with a better g value
            for open_node in open_list:
                if neighbor == open_node and neighbor.g >= open_node.g:
                    break
            else:
                # Add to open list if not already there with better g
                heapq.heappush(open_list, neighbor)

    # If we're here, no path was found
    return None


def create_voxel_grid(existing_paths, min_separation, cube_bounds, resolution):
    """Create a voxel grid representing occupied space around existing paths."""
    min_x, max_x, min_y, max_y, min_z, max_z = cube_bounds

    # Calculate grid dimensions
    x_size = int((max_x - min_x) / resolution) + 1
    y_size = int((max_y - min_y) / resolution) + 1
    z_size = int((max_z - min_z) / resolution) + 1

    # Initialize empty grid
    grid = np.zeros((x_size, y_size, z_size), dtype=bool)

    # Mark occupied voxels around each path
    buffer_cells = max(1, int(min_separation / resolution))

    for path in existing_paths:
        for x, y, z in path:
            # Convert to grid coordinates
            grid_x = int((x - min_x) / resolution)
            grid_y = int((y - min_y) / resolution)
            grid_z = int((z - min_z) / resolution)

            # Mark buffer zone around the path point
            for dx in range(-buffer_cells, buffer_cells + 1):
                for dy in range(-buffer_cells, buffer_cells + 1):
                    for dz in range(-buffer_cells, buffer_cells + 1):
                        nx, ny, nz = grid_x + dx, grid_y + dy, grid_z + dz

                        # Check if within grid bounds
                        if 0 <= nx < x_size and 0 <= ny < y_size and 0 <= nz < z_size:
                            # Check if within min_separation distance
                            dist = math.sqrt(dx**2 + dy**2 + dz**2) * resolution
                            if dist <= min_separation:
                                grid[nx, ny, nz] = True

    return {
        'grid': grid,
        'min_coords': (min_x, min_y, min_z),
        'resolution': resolution
    }


##################### Helper Methods ##################################################

def get_neighbors(node, resolution, cube_bounds):
    """Get all valid neighbors for a node within the cube bounds."""
    x, y, z = node.position
    neighbors = []

    # Define possible movements (6-connectivity in 3D space)
    movements = [
        (resolution, 0, 0), (-resolution, 0, 0),
        (0, resolution, 0), (0, -resolution, 0),
        (0, 0, resolution), (0, 0, -resolution)
    ]

    # Add diagonal movements (26-connectivity in 3D space)
    for dx in [-resolution, 0, resolution]:
        for dy in [-resolution, 0, resolution]:
            for dz in [-resolution, 0, resolution]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the center (current position)
                movements.append((dx, dy, dz))

    for dx, dy, dz in movements:
        new_pos = (x + dx, y + dy, z + dz)

        # Check if the new position is within the cube bounds
        min_x, max_x, min_y, max_y, min_z, max_z = cube_bounds
        if (min_x <= new_pos[0] <= max_x and
            min_y <= new_pos[1] <= max_y and
            min_z <= new_pos[2] <= max_z):
            neighbors.append(Node(new_pos, node))

    return neighbors

def is_position_occupied(position, voxel_grid, resolution):
    """Check if a position is occupied in the voxel grid."""
    grid = voxel_grid['grid']
    min_x, min_y, min_z = voxel_grid['min_coords']

    # Convert position to grid indices
    x_idx = int((position[0] - min_x) / resolution)
    y_idx = int((position[1] - min_y) / resolution)
    z_idx = int((position[2] - min_z) / resolution)

    # Check if indices are within grid bounds
    if (0 <= x_idx < grid.shape[0] and
        0 <= y_idx < grid.shape[1] and
        0 <= z_idx < grid.shape[2]):
        return grid[x_idx, y_idx, z_idx]

    return False

def is_path_valid(new_path, existing_paths, min_separation):
    """Check if new_path maintains minimum separation from all existing paths."""
    for path in existing_paths:
        if path_distance(new_path, path) < min_separation:
            return False
    return True

def path_distance(path1, path2):
    """Calculate the minimum distance between two paths."""
    min_dist = float('inf')

    # Simplify computation by checking a subset of points along each path
    # For a more accurate calculation, you could check all points or line segments
    for point1 in path1[::5]:  # Sample every 5th point
        for point2 in path2[::5]:
            dist = octile_distance(point1, point2)
            min_dist = min(min_dist, dist)

    return min_dist


def octile_distance(a, b):
    dx, dy, dz = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
    return max(dx, dy, dz) + (math.sqrt(2) - 1) * (dx + dy + dz - max(dx, dy, dz))

##################### Visualize ##################################################

def visualize_paths(paths, connections, cube_dimensions):
    """Visualize all paths within the cube."""
    width, height, depth = cube_dimensions

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cube boundaries
    r = [-width/2, width/2]
    for s, e in combinations(np.array(list(product(r, r, [-height/2, height/2]))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="black", alpha=0.3)

    # Plot paths
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))

    for i, path in enumerate(paths):
        xs, ys, zs = zip(*path)
        ax.plot(xs, ys, zs, color=colors[i], linewidth=2)

        # Mark start and end points
        start, end = connections[i]
        ax.scatter(*start, color=colors[i], marker='o', s=50)
        ax.scatter(*end, color=colors[i], marker='s', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{len(paths)} Conductive Paths in 3D Cube')

    # Set equal aspect ratio
    max_range = max(width, depth, height)
    ax.set_xlim(-max_range/2, max_range/2)
    ax.set_ylim(-max_range/2, max_range/2)
    ax.set_zlim(-height/2, height/2)

    plt.tight_layout()
    plt.show()

# tube smoothing
def smooth_path(path, existing_paths, min_separation, iterations=10):
    """Smooth a path to make it more natural using a simple relaxation algorithm."""
    if len(path) <= 2:
        return path  # Can't smooth a path with just start and end

    smoothed = list(path)

    for _ in range(iterations):
        # Don't modify first and last points (fixed endpoints)
        for i in range(1, len(smoothed) - 1):
            # Get adjacent points
            prev = smoothed[i - 1]
            next_pt = smoothed[i + 1]

            # Calculate new position by averaging with neighbors
            new_pos = tuple((prev[j] + next_pt[j]) / 2 for j in range(3))

            # Create a temporary path with the modified point
            temp_path = smoothed.copy()
            temp_path[i] = new_pos

            # Check if modified path is valid
            if is_path_valid(temp_path, [p for p in existing_paths if p != path], min_separation):
                smoothed[i] = new_pos

    return smoothed

def plot_voxel_grid_3d_plotly(voxel_grid):
    """Interactive 3D visualization of the voxel grid using Plotly."""

    grid = voxel_grid['grid']
    min_x, min_y, min_z = voxel_grid['min_coords']
    resolution = voxel_grid['resolution']

    # Get occupied voxel coordinates
    occupied_voxels = np.argwhere(grid)
    if len(occupied_voxels) == 0:
        print("No occupied voxels to display.")
        return

    # Convert indices to real-world coordinates
    x_vals, y_vals, z_vals = [], [], []
    for voxel in occupied_voxels:
        x, y, z = voxel * resolution + np.array([min_x, min_y, min_z])
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    # Create 3D scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers',
        marker=dict(size=resolution*20, color='red', opacity=0.5),
        name="Occupied Voxels"
    ))

    # Layout settings
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode="cube"
        ),
        title="Interactive 3D Voxel Grid",
    )

    fig.show()

##################### Export ##################################################

def export_openscad(paths, filename):
    """Exports paths as OpenSCAD polyline code."""
    with open(filename, "w") as f:
        f.write("module secure_tubes_polyline() {\n")
        for idx, path in enumerate(paths):
            points_str = ", ".join(f"[{x}, {y}, {z}]" for x, y, z in path)
            f.write(f"  polyline([{points_str}], thickness=tube_radius);\n")
        f.write("}\n")
    print(f"✅ OpenSCAD code saved to {filename}")