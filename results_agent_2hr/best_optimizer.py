
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
# Thrust coefficient (corrected to 26 elements)
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

# --- Helper functions for initial layout generation ---
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
    def edge_distances_for_all_points(i: int) -> jnp.ndarray:
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n_vertices]
        return _signed_distance_to_edge_line(px, py, x1, y1, x2, y2)

    all_distances = jax.vmap(edge_distances_for_all_points)(jnp.arange(n_vertices))
    min_distances_per_point = jnp.min(all_distances, axis=0)
    return min_distances_per_point > -threshold

def generate_initial_layout_hybrid(n_target, boundary, min_spacing, init_x_fallback, init_y_fallback, key,
                                   perturb_from_x=None, perturb_from_y=None, perturbation_strength=0.1):
    if perturb_from_x is not None and perturb_from_y is not None:
        # Perturb from the best known layout
        noise_x = jax.random.normal(key, shape=(n_target,)) * perturbation_strength * D
        noise_y = jax.random.normal(key, shape=(n_target,)) * perturbation_strength * D
        initial_x = perturb_from_x + noise_x
        initial_y = perturb_from_y + noise_y
    else:
        # Generate grid-based initial layout
        polygon_area_val = polygon_area(boundary)
        target_square_area_per_turbine = polygon_area_val / n_target
        approx_grid_spacing = jnp.maximum(min_spacing * 1.1, jnp.sqrt(target_square_area_per_turbine) * 0.9)

        min_x_b, min_y_b = jnp.min(boundary, axis=0)
        max_x_b, max_y_b = jnp.max(boundary, axis=0)

        grid_x_coords = jnp.arange(min_x_b, max_x_b, approx_grid_spacing)
        grid_y_coords = jnp.arange(min_y_b, max_y_b, approx_grid_spacing)

        mesh_x, mesh_y = jnp.meshgrid(grid_x_coords, grid_y_coords)
        candidate_x = mesh_x.flatten()
        candidate_y = mesh_y.flatten()

        inside_mask = is_inside_convex_polygon_batch(candidate_x, candidate_y, boundary)
        initial_x_grid = candidate_x[inside_mask]
        initial_y_grid = candidate_y[inside_mask]

        if len(initial_x_grid) > n_target:
            indices = jax.random.permutation(key, jnp.arange(len(initial_x_grid)))[:n_target]
            initial_x = initial_x_grid[indices]
            initial_y = initial_y_grid[indices]
        elif len(initial_x_grid) < n_target:
            initial_x = init_x_fallback
            initial_y = init_y_fallback
        else:
            initial_x = initial_x_grid
            initial_y = initial_y_grid
    
    return jnp.array(initial_x), jnp.array(initial_y)

# --- Define Objective Functions (global for sim, wd, ws, weights, boundary, min_spacing) ---
HOURS_PER_YEAR = 8760
GWH_CONVERSION_FACTOR = 1e-9 # Conversion from kW to GW (1e-6 for MW, 1e-9 for GW)

def objective_aep(x, y):
    r = sim(x, y, ws_amb=ws, wd_amb=wd, ti_amb=None)
    total_power_per_condition = jnp.sum(r.power()[:, :len(x)], axis=1)
    weighted_total_power_kW = jnp.sum(total_power_per_condition * weights)
    aep_gwh = weighted_total_power_kW * HOURS_PER_YEAR * GWH_CONVERSION_FACTOR
    return -aep_gwh

def objective_feasibility(x, y):
    b_penalty = boundary_penalty(x, y, boundary)
    s_penalty = spacing_penalty(x, y, min_spacing)
    return b_penalty + s_penalty

# --- Optimization Function for a Single Run ---
def run_single_optimization(initial_x, initial_y, boundary, min_spacing):
    # Stage 1: Feasibility Optimization
    settings_feasibility = SGDSettings(
        learning_rate=30.0,
        max_iter=1200,
        additional_constant_lr_iterations=150,
        gamma_min_factor=0.05,
        ks_rho=750.0,
        spacing_weight=50.0,
        boundary_weight=50.0,
        tol=5e-5
    )
    feasible_x, feasible_y = topfarm_sgd_solve(
        objective_feasibility, initial_x, initial_y,
        boundary, min_spacing, settings_feasibility
    )

    # Stage 2: AEP Optimization with adjusted beta parameters
    settings_aep = SGDSettings(
        learning_rate=12.0, 
        max_iter=3000,
        additional_constant_lr_iterations=250,
        gamma_min_factor=0.005,
        ks_rho=150.0,
        spacing_weight=1.0,
        boundary_weight=1.0,
        tol=1e-6,
        beta1=0.95, 
        beta2=0.99 
    )
    opt_x, opt_y = topfarm_sgd_solve(
        objective_aep, feasible_x, feasible_y,
        boundary, min_spacing, settings_aep
    )
    
    final_aep = -objective_aep(opt_x, opt_y) # Calculate positive AEP
    return opt_x, opt_y, final_aep

# --- Multi-Start Optimization Loop ---
num_starts = 20 
num_grid_starts = num_starts // 2
perturbation_strength = 0.1 # 0.1 * D for perturbation
best_aep = -jnp.inf
best_opt_x = None
best_opt_y = None

print(f"Starting hybrid multi-start optimization with {num_starts} runs...")

for i in range(num_starts):
    print(f"Run {i+1}/{num_starts}...")
    key = jax.random.PRNGKey(i) 

    if i < num_grid_starts:
        # Initial runs: grid-based layouts
        current_initial_x, current_initial_y = generate_initial_layout_hybrid(
            n_target, boundary, min_spacing, init_x_provided, init_y_provided, key
        )
    else:
        # Subsequent runs: perturb from the best found so far
        if best_opt_x is not None and best_opt_y is not None:
            current_initial_x, current_initial_y = generate_initial_layout_hybrid(
                n_target, boundary, min_spacing, init_x_provided, init_y_provided, key,
                perturb_from_x=best_opt_x, perturb_from_y=best_opt_y, perturbation_strength=perturbation_strength
            )
        else:
            # Fallback if no best_opt_x/y yet (shouldn't happen if num_grid_starts > 0)
            print("Warning: No best layout to perturb from, falling back to grid generation.")
            current_initial_x, current_initial_y = generate_initial_layout_hybrid(
                n_target, boundary, min_spacing, init_x_provided, init_y_provided, key
            )
    
    current_opt_x, current_opt_y, current_aep = run_single_optimization(
        current_initial_x, current_initial_y, boundary, min_spacing
    )
    
    print(f"Run {i+1} AEP: {current_aep:.2f} GWh")
    if current_aep > best_aep:
        best_aep = current_aep
        best_opt_x = current_opt_x
        best_opt_y = current_opt_y
        print(f"New best AEP: {best_aep:.2f} GWh")

print(f"\nHybrid multi-start optimization complete. Best AEP found: {best_aep:.2f} GWh")

# --- Output Results ---
with open(os.environ["FUNWAKE_OUTPUT"], "w") as f:
    json.dump({"x": [float(v) for v in best_opt_x],
               "y": [float(v) for v in best_opt_y]}, f)
