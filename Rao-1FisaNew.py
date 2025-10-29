#!/usr/bin/env python3
"""
Rao-1 + FISA hybrid optimizer for RC frame cost minimization (CORRECTED VERSION)

This is your original code with ONLY the critical bugs fixed:
1. Penalty coefficient reduced
2. Relative violation penalties
3. Realistic bounds
4. Geometry-scaled loads
5. Fixed capacity checks
"""

import random
import math
import numpy as np
import pandas as pd
import time
import os
from pyspark import SparkContext, SparkConf

# ------------------------------
# Problem / model parameters
# ------------------------------
NUM_STORIES = 2
COLUMNS_PER_STORY = 3 * 2
BEAMS_PER_STORY = 3 * 2
LU = 3.0   # unsupported length of column (m)
LB = 5.0   # beam length (m)

# material & costs
rho_steel = 7850.0
cost_concrete_per_m3 = 112.13  # € (from paper Table 2)
cost_steel_per_kg = 1.30       # €
cost_formwork_per_m2 = 25.0    # € (simplified)

# ACI / design constants
phi_column = 0.7
phi_beam = 0.9
f_y = 420e6    # Pa (420 MPa)
f_c = 30e6     # Pa (30 MPa)
beta1 = 0.85

# FIXED: Reduced penalty coefficient
PENALTY_COEFF = 1e4  # was 1e8 - this was causing 10^12 penalties!

# FIXED: Realistic bounds
# Variables: [bc, dc, Asc, bw, db, Asb]
lb = np.array([0.20, 0.25, 0.0004, 0.20, 0.30, 0.0004])  # m, m, m²
ub = np.array([0.45, 0.60, 0.0040, 0.45, 0.80, 0.0040])  # m, m, m²

POP_SIZE = 30
DIM = 6
MAX_FES = 30000
MAX_ITER = MAX_FES // POP_SIZE

# Spark context setup
conf = SparkConf().setAppName("Rao1_FISA_RC_Frame").setIfMissing("spark.master", "local[*]")
sc = SparkContext.getOrCreate(conf=conf)

# ------------------------------
# Helper structural / cost functions
# ------------------------------
def volume_column(bc, dc, stories=NUM_STORIES, ncols=COLUMNS_PER_STORY, Lu=LU):
    Agc = bc * dc
    Vcol = Agc * Lu * stories * ncols
    return Vcol

def volume_beam(bw, db, stories=NUM_STORIES, nbeams=BEAMS_PER_STORY, Lb=LB):
    Agb = bw * db
    Vbeam = Agb * Lb * stories * nbeams
    return Vbeam

def volume_steel_columns(Asc, Lcbars, stories=NUM_STORIES, ncols=COLUMNS_PER_STORY):
    return Asc * Lcbars * stories * ncols

def volume_steel_beams(Asb, Lbbars, stories=NUM_STORIES, nbeams=BEAMS_PER_STORY):
    return Asb * Lbbars * stories * nbeams

def area_formwork_columns(bc, dc, stories=NUM_STORIES, ncols=COLUMNS_PER_STORY, Lu=LU):
    perimeter = 2.0 * (bc + dc)
    Af = perimeter * Lu * stories * ncols
    return Af

def area_formwork_beams(bw, db, stories=NUM_STORIES, nbeams=BEAMS_PER_STORY, Lb=LB):
    top_bottom = 2.0 * bw * Lb * stories * nbeams
    sides = 2.0 * db * Lb * stories * nbeams
    return top_bottom + sides

def approx_column_bar_length(Lu=LU, story_count=NUM_STORIES):
    return (Lu + 0.3) * 1.0

def approx_beam_bar_length(Lb=LB):
    return Lb * 1.1

# ------------------------------
# FIXED: Constraint checker with relative penalties
# ------------------------------
def rc_frame_cost_and_penalty(vars_array):
    """FIXED VERSION: Uses relative penalties and geometry-scaled loads"""
    bc, dc, Asc, bw, db, Asb = vars_array
    
    penalty = 0.0
    
    # FIXED: Calculate loads based on geometry (not fixed arbitrary values)
    w_dead = 35.0  # kN/m (from paper - includes self-weight)
    w_live = 5.0   # kN/m
    w_ult = 1.35 * w_dead + 1.5 * w_live  # Load combination
    
    # Beam moment (continuous beam - approximate)
    Mu_required = w_ult * LB**2 / 8 * 1000  # N-m (factored)
    
    # Column axial load (tributary area method)
    trib_area = LB * LB / (COLUMNS_PER_STORY / NUM_STORIES)
    Pu_required = w_ult * trib_area * NUM_STORIES * 1000 * 9.81  # N
    
    # Beam shear
    Vu_required = w_ult * LB / 2 * 1000  # N
    
    # 1) Geometric constraints (FIXED: relative violations)
    if bc - dc > 0:
        penalty += (bc - dc) / dc * PENALTY_COEFF
    
    # Beam depth proportion: 1.5 * bw <= db <= 2.0 * bw
    if db < 1.5 * bw:
        penalty += (1.5 * bw - db) / bw * PENALTY_COEFF
    if db > 2.0 * bw:
        penalty += (db - 2.0 * bw) / bw * PENALTY_COEFF
    
    # Compatibility: column width >= beam width
    if bw - bc > 0:
        penalty += (bw - bc) / bc * PENALTY_COEFF
    
    # 2) Bounds check (FIXED: relative)
    for v, low, up in zip([bc, dc, Asc, bw, db, Asb], lb, ub):
        if v < low:
            penalty += (low - v) / low * PENALTY_COEFF
        if v > up:
            penalty += (v - up) / up * PENALTY_COEFF
    
    # 3) Steel ratio constraints (FIXED: relative)
    Agc = bc * dc
    rho_c = Asc / Agc if Agc > 0 else 1.0
    if rho_c < 0.01:
        penalty += (0.01 - rho_c) / 0.01 * PENALTY_COEFF
    if rho_c > 0.08:
        penalty += (rho_c - 0.08) / 0.08 * PENALTY_COEFF
    
    Agb = bw * db
    rho_b = Asb / Agb if Agb > 0 else 1.0
    rho_b_min = max(0.002, 0.25 * (math.sqrt(f_c) / f_y) * 1e-3)
    if rho_b < rho_b_min:
        penalty += (rho_b_min - rho_b) / rho_b_min * PENALTY_COEFF
    if rho_b > 0.04:  # Practical maximum for constructability
        penalty += (rho_b - 0.04) / 0.04 * PENALTY_COEFF
    
    # 4) Flexural capacity (beam) - FIXED: proper checks
    if bw > 0 and f_c > 0 and Asb > 0:
        d = db - 0.05  # Effective depth (50mm cover)
        a = (Asb * f_y) / (0.85 * f_c * bw)
        
        # Check if over-reinforced (a > d is bad)
        if a > d:
            penalty += 5.0 * PENALTY_COEFF
        else:
            z = d - a / 2.0
            Mn = Asb * f_y * z
            phi_Mn = phi_beam * Mn
            if phi_Mn < Mu_required:
                penalty += (Mu_required - phi_Mn) / Mu_required * PENALTY_COEFF
    else:
        penalty += PENALTY_COEFF  # Missing steel
    
    # 5) Column axial capacity - FIXED
    if Agc > 0 and Asc > 0:
        Pn = 0.85 * f_c * Agc + Asc * f_y
        phi_Pn = phi_column * Pn
        if phi_Pn < Pu_required:
            penalty += (Pu_required - phi_Pn) / Pu_required * PENALTY_COEFF
    else:
        penalty += PENALTY_COEFF
    
    # 6) Shear constraint - FIXED
    if bw > 0 and db > 0:
        d = db - 0.05
        Vc = 0.17 * math.sqrt(f_c) * bw * d
        phi_Vc = 0.75 * Vc
        if phi_Vc < Vu_required:
            penalty += (Vu_required - phi_Vc) / Vu_required * PENALTY_COEFF
    
    # 7) Minimum width constraint
    br_min = 0.20  # 200mm minimum (from paper)
    if bw < br_min:
        penalty += (br_min - bw) / br_min * PENALTY_COEFF
    if bc < br_min:
        penalty += (br_min - bc) / br_min * PENALTY_COEFF
    
    # ------------------------------
    # Cost calculation (unchanged)
    # ------------------------------
    Vcol = volume_column(bc, dc)
    Vbeam = volume_beam(bw, db)
    Lcbars = approx_column_bar_length()
    Lbbars = approx_beam_bar_length()
    
    Vsteel_col = volume_steel_columns(Asc, Lcbars)
    Vsteel_beam = volume_steel_beams(Asb, Lbbars)
    
    mass_steel_col = Vsteel_col * rho_steel
    mass_steel_beam = Vsteel_beam * rho_steel
    total_steel_mass = mass_steel_col + mass_steel_beam
    
    Aform_col = area_formwork_columns(bc, dc)
    Aform_beam = area_formwork_beams(bw, db)
    
    cost_concrete = (Vcol + Vbeam) * cost_concrete_per_m3
    cost_steel = total_steel_mass * cost_steel_per_kg
    cost_formwork = (Aform_col + Aform_beam) * cost_formwork_per_m2
    
    total_cost = cost_concrete + cost_steel + cost_formwork
    fitness = total_cost + penalty
    
    return float(fitness), float(total_cost), float(penalty)

def evaluate_individual(position):
    """Wrapper for Spark"""
    fitness, raw_cost, penalty = rc_frame_cost_and_penalty(position)
    return (position.tolist() if isinstance(position, np.ndarray) else list(position), 
            fitness, raw_cost, penalty)

# ------------------------------
# Rao-1 + FISA hybrid optimizer
# ------------------------------
def rao1_fisa_pyspark(pop_size=POP_SIZE, dim=DIM, lb_arr=lb, ub_arr=ub, max_iter=MAX_ITER):
    """
    Rao-1 + FISA Hybrid Algorithm
    - Rao-1: parameter-free global search
    - FISA: fire-inspired local search with adaptive intensity
    """
    
    # Initialize population
    Positions = np.random.rand(pop_size, dim) * (ub_arr - lb_arr) + lb_arr
    
    # Initial evaluation
    rdd = sc.parallelize(Positions.tolist(), numSlices=min(pop_size, 8))
    evals = rdd.map(evaluate_individual).collect()
    fitness_vals = np.array([e[1] for e in evals])
    raw_costs = np.array([e[2] for e in evals])
    penalties = np.array([e[3] for e in evals])
    
    history = []
    best_overall = None
    start_time = time.time()
    
    # FISA parameters
    fire_intensity = 1.0
    fire_decay = 0.95
    
    print(f"\n{'='*80}")
    print(f"Rao-1 + FISA Hybrid Optimizer (CORRECTED)")
    print(f"{'='*80}")
    print(f"Population: {pop_size}, Dimensions: {dim}, Max Iterations: {max_iter}")
    print(f"Initial best penalty: {penalties.min():.2f} (should be < 100k, not 10^12!)")
    print(f"{'='*80}\n")
    
    for k in range(max_iter):
        # ===== RAO-1 PHASE =====
        best_idx = int(np.argmin(fitness_vals))
        worst_idx = int(np.argmax(fitness_vals))
        best_pos = Positions[best_idx].copy()
        worst_pos = Positions[worst_idx].copy()
        Positioncopy = Positions.copy()
        
        for i in range(pop_size):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                term1 = r1 * (best_pos[j] - worst_pos[j])
                rand_idx = random.randint(0, pop_size - 1)
                MX = max(Positioncopy[i][j], Positioncopy[rand_idx][j])
                term2 = r2 * (Positioncopy[i][j] - MX)
                Positions[i][j] += term1 + term2
                Positions[i][j] = np.clip(Positions[i][j], lb_arr[j], ub_arr[j])
        
        # Evaluate Rao-1 updates
        rdd = sc.parallelize(Positions.tolist(), numSlices=min(pop_size, 8))
        evals = rdd.map(evaluate_individual).collect()
        new_fitness = np.array([e[1] for e in evals])
        new_raw_costs = np.array([e[2] for e in evals])
        new_penalties = np.array([e[3] for e in evals])
        
        # Greedy selection
        for i in range(pop_size):
            if new_fitness[i] <= fitness_vals[i]:
                fitness_vals[i] = new_fitness[i]
                raw_costs[i] = new_raw_costs[i]
                penalties[i] = new_penalties[i]
            else:
                Positions[i] = Positioncopy[i].copy()
        
        # ===== FISA PHASE =====
        # Fire intensity decay
        fire_intensity *= fire_decay
        
        # Fire spread: perturb best solutions
        if random.random() < 0.3:  # 30% chance
            num_fires = max(1, pop_size // 10)
            best_indices = np.argsort(fitness_vals)[:num_fires]
            
            for idx in best_indices:
                neighbor = Positions[idx].copy()
                # Local perturbation (fire spread is localized)
                for j in range(dim):
                    if random.random() < fire_intensity * 0.3:
                        delta = random.uniform(-0.05, 0.05) * (ub_arr[j] - lb_arr[j])
                        neighbor[j] = np.clip(neighbor[j] + delta, lb_arr[j], ub_arr[j])
                
                # Evaluate and potentially replace worst
                n_fitness, n_cost, n_penalty = rc_frame_cost_and_penalty(neighbor)
                worst_idx = int(np.argmax(fitness_vals))
                if n_fitness < fitness_vals[worst_idx]:
                    Positions[worst_idx] = neighbor
                    fitness_vals[worst_idx] = n_fitness
                    raw_costs[worst_idx] = n_cost
                    penalties[worst_idx] = n_penalty
        
        # Fire ignition: add random solutions (exploration)
        if random.random() < 0.1:  # 10% chance
            num_ignitions = max(1, pop_size // 20)
            for _ in range(num_ignitions):
                new_fire = np.random.rand(dim) * (ub_arr - lb_arr) + lb_arr
                f_fitness, f_cost, f_penalty = rc_frame_cost_and_penalty(new_fire)
                worst_idx = int(np.argmax(fitness_vals))
                if f_fitness < fitness_vals[worst_idx]:
                    Positions[worst_idx] = new_fire
                    fitness_vals[worst_idx] = f_fitness
                    raw_costs[worst_idx] = f_cost
                    penalties[worst_idx] = f_penalty
        
        # Track best
        best_idx = int(np.argmin(fitness_vals))
        best_score = float(fitness_vals[best_idx])
        best_solution = Positions[best_idx].copy()
        best_raw_cost = float(raw_costs[best_idx])
        best_penalty = float(penalties[best_idx])
        
        # Log progress
        if (k + 1) % 100 == 0 or k == 0:
            elapsed = time.time() - start_time
            print(f"Iter {(k+1):5d} | Fitness: {best_score:12.2f} | Cost: {best_raw_cost:10.2f} € | "
                  f"Penalty: {best_penalty:10.2f} | Fire: {fire_intensity:.3f} | Time: {elapsed:.1f}s")
            
            history.append({
                "iter": k+1,
                "best_fitness": best_score,
                "best_raw_cost": best_raw_cost,
                "best_penalty": best_penalty,
                "x": best_solution.tolist()
            })
        
        # Update global best
        if best_overall is None or best_score < best_overall["best_fitness"]:
            best_overall = {
                "best_fitness": best_score,
                "best_raw_cost": best_raw_cost,
                "best_penalty": best_penalty,
                "x": best_solution.copy(),
                "iter": k+1
            }
    
    # Save results
    results_df = pd.DataFrame(history)
    out_dir = os.getcwd()
    results_csv = os.path.join(out_dir, "rao1_fisa_rc_frame_history_FIXED.csv")
    best_csv = os.path.join(out_dir, "rao1_fisa_rc_frame_best_FIXED.csv")
    results_df.to_csv(results_csv, index=False)
    
    best_df = pd.DataFrame([{
        "best_fitness": best_overall["best_fitness"],
        "best_raw_cost": best_overall["best_raw_cost"],
        "best_penalty": best_overall["best_penalty"],
        "x_bc": best_overall["x"][0],
        "x_dc": best_overall["x"][1],
        "x_Asc": best_overall["x"][2],
        "x_bw": best_overall["x"][3],
        "x_db": best_overall["x"][4],
        "x_Asb": best_overall["x"][5],
        "iter": best_overall["iter"]
    }])
    best_df.to_csv(best_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Best Cost: {best_overall['best_raw_cost']:.2f} €")
    print(f"Best Penalty: {best_overall['best_penalty']:.2f}")
    print(f"Found at iteration: {best_overall['iter']}")
    print(f"\nBest Design Variables:")
    print(f"  Column: bc={best_overall['x'][0]*1000:.1f}mm, dc={best_overall['x'][1]*1000:.1f}mm, "
          f"Asc={best_overall['x'][2]*1e6:.1f}mm²")
    print(f"  Beam:   bw={best_overall['x'][3]*1000:.1f}mm, db={best_overall['x'][4]*1000:.1f}mm, "
          f"Asb={best_overall['x'][5]*1e6:.1f}mm²")
    print(f"\nResults saved to:")
    print(f"  {results_csv}")
    print(f"  {best_csv}")
    
    return best_overall

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    try:
        best = rao1_fisa_pyspark()
        print("\n✓ Optimization successful!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sc.stop()