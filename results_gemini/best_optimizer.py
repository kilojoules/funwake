import jax
jax.config.update("jax_enable_x64", True)
import os, json
import jax.numpy as jnp
import numpy as np
from pixwake import Curve, Turbine, WakeSimulation
from pixwake.deficit import BastankhahGaussianDeficit
from pixwake.optim.sgd import SGDSettings, topfarm_sgd_solve, boundary_penalty, spacing_penalty
from pixwake.optim.boundary import _signed_distance_to_edge_line # For initial layout generation

# Load problem data from environment variable
with open(os.environ["FUNWAKE_PROBLEM"]) as f:
    info = json.load(f)

# --- Setup Turbine and Wake Simulation ---
# IEA 15 MW turbine characteristics (D=240m, H=150m)
D = info["rotor_diameter"]
HUB_HEIGHT = 150.0  # Standard hub height for IEA 15 MW

# IEA 15 MW power and thrust coefficient curves (example data, assuming standard)
# Wind speeds in m/s
ws_arr = jnp.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25.0])
# Power in kW
power = jnp.array([0,0,2.399,209.258,689.198,1480.608,2661.238,4308.929,6501.057,9260.516,12081.404,13937.297,14705.016,14931.039,14985.209,14996.906,14999.343,14999.855,14999.966,14999.992,14999.998,14999.999,15000,15000,15000,15000.0])
# Thrust coefficient
ct = jnp.array([0.889,0.889,0.889,0.8,0.8,0.8,0.8,0.8,0.8,0.793,0.735,0.61,0.476,0.37,0.292,0.234,0.191,0.158,0.132,0.112,0.096,0.083,0.072,0.063,0.055,0.049])

# Create Turbine and WakeSimulation objects
turbine = Turbine(rotor_diameter=D, hub_height=HUB_HEIGHT,
                  power_curve=Curve(ws=ws_arr, values=power),
                  ct_curve=Curve(ws=ws_arr, values=ct))
sim = WakeSimulation(turbine, BastankhahGaussianDeficit(k=0.04))

# --- Load Problem-Specific Data ---
wd = jnp.array(info["wind_rose"]["directions_deg"])
ws = jnp.array(info["wind_rose"]["speeds_ms"])
weights = jnp.array(info["wind_rose"]["weights"])
boundary = jnp.array(info["boundary_vertices"])
init_x_provided = jnp.array(info["init_x"])
init_y_provided = jnp.array(info["init_y"])
min_spacing = info["min_spacing_m"]
n_target = len(init_x_provided) # Number of turbines

# --- Strategy: Smart Initial Layout Generation ---
# Instead of using the potentially poor `init_x`/`init_y` provided,
# generate a more suitable initial grid within the polygon boundary.

def polygon_area(vertices):
    """Calculates the area of a polygon using the shoelace formula."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * jnp.abs(jnp.dot(x, jnp.roll(y, 1)) - jnp.dot(y, jnp.roll(x, 1)))

def is_inside_convex_polygon_batch(px, py, vertices, threshold=1e-5):
    """
    Checks if a batch of points (px, py) are inside a convex polygon.
    Assumes CCW order for vertices.
    """
    n_vertices = vertices.shape[0]
    # Function to compute signed distance to a single edge for all points
    def edge_distances_for_all_points(i: int) -> jnp.ndarray:
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n_vertices]
        return _signed_distance_to_edge_line(px, py, x1, y1, x2, y2)

    # Vmap over edges to get distances for all points to all edges
    all_distances = jax.vmap(edge_distances_for_all_points)(jnp.arange(n_vertices)) # (n_edges, n_pts)
    
    # A point is inside if its minimum distance to any edge is positive (or near zero)
    min_distances_per_point = jnp.min(all_distances, axis=0) # (n_pts,)
    return min_distances_per_point > -threshold

print("Generating initial layout...")
# Calculate target grid spacing based on polygon area and number of turbines
polygon_area_val = polygon_area(boundary)
target_square_area_per_turbine = polygon_area_val / n_target
# Aim for a spacing slightly larger than min_spacing, but also related to density
approx_grid_spacing = jnp.maximum(min_spacing * 1.1, jnp.sqrt(target_square_area_per_turbine) * 0.9)

# Determine grid bounds from the polygon's bounding box
min_x_b, min_y_b = jnp.min(boundary, axis=0)
max_x_b, max_y_b = jnp.max(boundary, axis=0)

# Create a dense grid of candidate points
grid_x_coords = jnp.arange(min_x_b, max_x_b, approx_grid_spacing)
grid_y_coords = jnp.arange(min_y_b, max_y_b, approx_grid_spacing)

mesh_x, mesh_y = jnp.meshgrid(grid_x_coords, grid_y_coords)
candidate_x = mesh_x.flatten()
candidate_y = mesh_y.flatten()

# Filter candidates to keep only those inside the polygon
inside_mask = is_inside_convex_polygon_batch(candidate_x, candidate_y, boundary)
initial_x_grid = candidate_x[inside_mask]
initial_y_grid = candidate_y[inside_mask]

# Handle cases where the grid generation doesn't yield exactly n_target turbines
if len(initial_x_grid) > n_target:
    # Randomly select n_target points from the generated grid
    key = jax.random.PRNGKey(42) # Fixed seed for reproducibility
    indices = jax.random.permutation(key, jnp.arange(len(initial_x_grid)))[:n_target]
    initial_x = initial_x_grid[indices]
    initial_y = initial_y_grid[indices]
    print(f"Generated {len(initial_x_grid)} points, selected {n_target}.")
elif len(initial_x_grid) < n_target:
    # If not enough points, fall back to the provided initial layout and print a warning
    print(f"Warning: Generated only {len(initial_x_grid)} points, need {n_target}. Falling back to provided initial layout.")
    initial_x = init_x_provided
    initial_y = init_y_provided
else: # Exactly n_target points
    initial_x = initial_x_grid
    initial_y = initial_y_grid

# Ensure initial_x, initial_y are jnp.arrays
initial_x = jnp.array(initial_x)
initial_y = jnp.array(initial_y)

# --- Define Objective Functions ---

# AEP in GWh (Annual Energy Production)
HOURS_PER_YEAR = 8760
GWH_CONVERSION_FACTOR = 1e-9 # Conversion from kW to GW (1e-6 for MW, 1e-9 for GW)
def objective_aep(x, y):
    """
    Calculates the negative Annual Energy Production (AEP) in GWh.
    The optimizer minimizes this value to maximize AEP.
    """
    # Simulate wake effects and get power for each turbine under each wind condition
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    
    # Sum power across all turbines for each wind condition (kW)
    total_power_per_condition = jnp.sum(r.power()[:, :len(x)], axis=1) # Shape (n_conditions,)
    
    # Calculate weighted average power over all wind conditions (kW)
    weighted_total_power_kW = jnp.sum(total_power_per_condition * weights)
    
    # Convert to GWh: kW * hours/year * (GW/kW) = GWh
    aep_gwh = weighted_total_power_kW * HOURS_PER_YEAR * GWH_CONVERSION_FACTOR
    
    return -aep_gwh # Minimize negative AEP == Maximize AEP

def objective_feasibility(x, y):
    """
    Calculates the sum of boundary and spacing penalties.
    The optimizer minimizes this value to achieve a feasible layout.
    """
    b_penalty = boundary_penalty(x, y, boundary)
    s_penalty = spacing_penalty(x, y, min_spacing)
    return b_penalty + s_penalty

# --- Optimization Strategy: Two-Stage Approach with Refined Settings ---
# This approach first optimizes for feasibility and then for AEP,
# starting from a smarter initial layout.

# --- Stage 1: Feasibility Optimization ---
# Aim for quick convergence to a feasible layout with strong penalty enforcement.
print("Starting Stage 1: Feasibility Optimization...")
settings_feasibility = SGDSettings(
    learning_rate=30.0,  # Aggressive LR for quick movement
    max_iter=1200,       # Sufficient iterations to find a feasible region
    additional_constant_lr_iterations=150, # Initial constant LR phase
    gamma_min_factor=0.05, # Faster LR decay to settle faster
    ks_rho=750.0,        # Very strict KS aggregation for constraints
    spacing_weight=50.0, # Emphasize spacing constraint heavily
    boundary_weight=50.0,# Emphasize boundary constraint heavily
    tol=5e-5             # Slightly relaxed tolerance for initial feasibility
)

feasible_x, feasible_y = topfarm_sgd_solve(
    objective_feasibility, initial_x, initial_y, # Use the smart initial layout
    boundary, min_spacing, settings_feasibility
)
print("Stage 1 (Feasibility Optimization) complete.")

# --- Stage 2: AEP Optimization ---
# Use the feasible layout from Stage 1 as a starting point.
# Focus on maximizing AEP with a more refined optimization.
print("Starting Stage 2: AEP Optimization...")
settings_aep = SGDSettings(
    learning_rate=8.0,   # Moderate LR for fine-tuning AEP
    max_iter=3000,       # Longer optimization phase for better AEP
    additional_constant_lr_iterations=250, # Allow LR decay for most of the run
    gamma_min_factor=0.005, # Slower LR decay for better convergence to optimum
    ks_rho=150.0,        # Balanced KS aggregation, allowing internal alpha to scale
    spacing_weight=1.0,  # Default weights, rely on internal alpha for constraint strength
    boundary_weight=1.0,
    tol=1e-6             # Stricter tolerance for AEP convergence
)

opt_x, opt_y = topfarm_sgd_solve(
    objective_aep, feasible_x, feasible_y, # Start from the feasible layout
    boundary, min_spacing, settings_aep
)
print("Stage 2 (AEP Optimization) complete.")

# --- Output Results ---
# Write the optimized turbine coordinates to the specified output file
with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
    json.dump({"x": [float(v) for v in opt_x],
               "y": [float(v) for v in opt_y]}, f)