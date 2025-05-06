import math
import numpy as np
import matplotlib.pyplot as plt
import heapq


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


def find_all_paths(connections, cylinder_params, min_separation):
    """Find all collision-free paths for the given connections in a cylindrical shape."""
    radius_inner = cylinder_params['radius_inner']
    cylinder_height = cylinder_params['cylinder_height']
    resolution = cylinder_params['resolution']

    # Define cylinder bounds (as a bounding box)
    cylinder_bounds = (
    -radius_inner, radius_inner, -radius_inner, radius_inner, -cylinder_height / 2 +1 , cylinder_height / 2 -1)  # todo maybe adjust here

    paths = []
    failed_paths = []

    for i, (start, end) in enumerate(connections):
        print(f"Finding path {i + 1}/{len(connections)}: {start} to {end}")

        # Find path using A*
        path = astar(start, end, cylinder_bounds, cylinder_params, paths, min_separation, resolution)

        if path:
            paths.append(path)
            print(f"  Path found with {len(path)} points")
        else:
            failed_paths.append((start, end))
            print(f"  No valid path found")

    return paths, failed_paths


def astar(start, end, cylinder_bounds, cylinder_params, existing_paths, min_separation, resolution):
    """A* algorithm for 3D path finding with collision avoidance in a cylindrical shape."""
    # Create start and end nodes
    start_node = Node(start)
    end_node = Node(end)

    # Initialize open and closed lists
    open_list = []
    closed_set = set()

    # Add the start node to the open list
    heapq.heappush(open_list, start_node)

    # Create a voxel grid to track occupied space
    voxel_grid = create_voxel_grid(existing_paths, min_separation, cylinder_bounds, resolution)

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
        for neighbor in get_neighbors(current_node, resolution, cylinder_params):
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


def create_voxel_grid(existing_paths, min_separation, cylinder_bounds, resolution):
    """Create a voxel grid representing occupied space around existing paths."""
    min_x, max_x, min_y, max_y, min_z, max_z = cylinder_bounds

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
                            dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * resolution
                            if dist <= min_separation:
                                grid[nx, ny, nz] = True

    return {
        'grid': grid,
        'min_coords': (min_x, min_y, min_z),
        'resolution': resolution
    }


def get_neighbors(node, resolution, cylinder_params):
    """Get all valid neighbors for a node within the cylindrical bounds."""
    x, y, z = node.position
    neighbors = []

    # Define possible movements (26-connectivity in 3D space)
    movements = []
    for dx in [-resolution, 0, resolution]:
        for dy in [-resolution, 0, resolution]:
            for dz in [-resolution, 0, resolution]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the center (current position)
                movements.append((dx, dy, dz))

    for dx, dy, dz in movements:
        new_pos = (x + dx, y + dy, z + dz)

        # Check if the new position is within the cylindrical bounds
        if is_within_cylinder(new_pos, cylinder_params):
            neighbors.append(Node(new_pos, node))

    return neighbors


def is_within_cylinder(position, cylinder_params):
    """Check if a position is within the cylindrical shape."""
    x, y, z = position
    radius_inner = cylinder_params['radius_inner']
    cylinder_height = cylinder_params['cylinder_height']

    # Calculate distance from center axis
    distance_from_center = math.sqrt(x ** 2 + y ** 2)

    # Check if within height bounds
    within_height = -cylinder_height / 2 +1 <= z <= cylinder_height / 2 -1

    # Check if within radial bounds
    within_radius = distance_from_center <= radius_inner

    return within_height and within_radius


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
    """Calculate 3D octile distance between two points."""
    dx, dy, dz = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
    return max(dx, dy, dz) + (math.sqrt(2) - 1) * (dx + dy + dz - max(dx, dy, dz))


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


def visualize_paths(paths, connections, cylinder_params):
    """Visualize all paths within the cylinder."""
    radius_inner = cylinder_params['radius_inner']
    cylinder_height = cylinder_params['cylinder_height']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot cylinder outline
    u = np.linspace(0, 2 * np.pi, 30)
    h = np.linspace(-cylinder_height / 2, cylinder_height / 2, 10)
    x = np.outer(radius_inner * np.cos(u), np.ones(len(h)))
    y = np.outer(radius_inner * np.sin(u), np.ones(len(h)))
    z = np.outer(np.ones(len(u)), h)

    ax.plot_surface(x, y, z, color='gray', alpha=0.2)

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
    ax.set_title(f'{len(paths)} Conductive Paths in Cylinder')

    # Set equal aspect ratio
    max_range = max(radius_inner * 2, cylinder_height)
    ax.set_xlim(-max_range / 2, max_range / 2)
    ax.set_ylim(-max_range / 2, max_range / 2)
    ax.set_zlim(-cylinder_height / 2, cylinder_height / 2)

    plt.tight_layout()
    plt.show()


def export_openscad(paths, filename="cylindrical_paths.scad"):
    """Exports paths as OpenSCAD polyline code."""
    with open(filename, "w") as f:
        f.write("// Generated paths for cylindrical design\n")
        f.write("module secure_tubes_polyline() {\n")
        for idx, path in enumerate(paths):
            f.write(f"    // Path {idx + 1}\n")
            points_str = ", ".join(f"[{x}, {y}, {z}]" for x, y, z in path)
            f.write(f"    polyline([{points_str}], thickness=tube_radius);\n")
        f.write("}\n")
    print(f"✅ OpenSCAD code saved to {filename}")


def generate_cylinder_points(cylinder_params):
    """Generate the connection points for the cylindrical design."""
    num_cylinders = cylinder_params['num_cylinders']
    cylinder_circle_radius = cylinder_params['cylinder_circle_radius']
    cylinder_height = cylinder_params['cylinder_height']
    resolution = cylinder_params['resolution']

    top_list = []
    bottom_list = []

    for i in range(num_cylinders):
        angle = i * (360 / num_cylinders)
        angle_rad = math.radians(angle)

        x = cylinder_circle_radius * math.cos(angle_rad)
        y = cylinder_circle_radius * math.sin(angle_rad)

        # Add top point
        top_list.append((x, y, cylinder_height / 2 - resolution))

        # Add bottom point
        bottom_list.append((x, y, -cylinder_height / 2 + resolution))

    return top_list, bottom_list
