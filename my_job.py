#!/usr/bin/env python3
"""
Rao-1 + FISA hybrid optimizer for RC frame cost minimization (PySpark version)

Save as: rao1_fisa_rc_frame_pyspark.py
Run with: spark-submit rao1_fisa_rc_frame_pyspark.py

This implementation is based on the RC frame problem statement you uploaded (objective = cost of concrete, steel, formwork; design variables: beam/column widths, depths, steel areas; many ACI constraints). See uploaded file for full formulations. :contentReference[oaicite:4]{index=4}
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
# (use values consistent with the paper / model described)
# ------------------------------
# geometry/structure (three-bay, two-story example from paper)
NUM_STORIES = 2
COLUMNS_PER_STORY = 3 * 2  # example: 3 bays x 2 directions -> adjust if needed
BEAMS_PER_STORY = 3 * 2    # approximate (adjust to your model)
# Use the paper's sample geometry or override here:
LU = 3.0   # unsupported length (clear height) of column per story (m) - adjust
LB = 5.0   # beam length between column center lines (m) - typical span - adjust

# material & costs (units: m, m^2, m^3, kg)
rho_steel = 7850.0        # kg/m^3 (steel density)
cost_concrete_per_m3 = 100.0   # example: currency units per m^3 - adjust
cost_steel_per_kg = 1.0        # currency per kg - adjust
cost_formwork_per_m2 = 10.0    # currency per m^2 - adjust

# ACI / design constants (approx)
phi_column = 0.7      # strength reduction factor for tied columns
phi_beam = 0.9        # for beams (as referenced)
f_y = 420e6           # steel yield (Pa) -> 420 MPa
f_c = 30e6            # concrete strength (Pa) - choose 25e6..30e6 if desired
beta1 = 0.85          # as in paper (for f'c < 4000 psi); adjust if needed

# Penalty multiplier for constraint violations
PENALTY_COEFF = 1e8

# Design variable bounds (practical/architectural bounds)
# Variables: [bc, dc, Asc, bw, db, Asb]
lb = np.array([0.22, 0.3, 0.0001, 0.22, 0.3, 0.0001])  # lower bounds (m, m, m^2)
ub = np.array([0.45, 0.6, 0.02, 0.45, 0.8, 0.02])     # upper bounds

# Derived convenience
POP_SIZE = 30
DIM = 6
MAX_FES = 30000
MAX_ITER = MAX_FES // POP_SIZE

# Spark context setup (for local testing change master to "local[*]")
conf = SparkConf().setAppName("Rao1_FISA_RC_Frame").setIfMissing("spark.master", "local[*]")
sc = SparkContext(conf=conf)

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
    # Asc is total area of longitudinal steel per column (m^2)
    return Asc * Lcbars * stories * ncols

def volume_steel_beams(Asb, Lbbars, stories=NUM_STORIES, nbeams=BEAMS_PER_STORY):
    return Asb * Lbbars * stories * nbeams

def area_formwork_columns(bc, dc, stories=NUM_STORIES, ncols=COLUMNS_PER_STORY, Lu=LU):
    # approximate formwork surface area for column (4 faces)
    perimeter = 2.0 * (bc + dc)
    Af = perimeter * Lu * stories * ncols
    return Af

def area_formwork_beams(bw, db, stories=NUM_STORIES, nbeams=BEAMS_PER_STORY, Lb=LB):
    # approximate beam formwork area (top + bottom + 2 sides)
    top_bottom = 2.0 * bw * Lb * stories * nbeams
    sides = 2.0 * db * Lb * stories * nbeams
    return top_bottom + sides

# lengths for bars - approximate (user used PROKON for exact bar layouts)
def approx_column_bar_length(Lu=LU, story_count=NUM_STORIES):
    # total length of longitudinal bars in single column (approx)
    return (Lu + 0.3) * 1.0  # small extra for anchorage; will be multiplied later

def approx_beam_bar_length(Lb=LB):
    return Lb * 1.1  # small extra for anchorage

# ------------------------------
# Constraint checker & penalty-based cost function
# ------------------------------
def rc_frame_cost_and_penalty(vars_array):
    """vars_array: array-like [bc, dc, Asc, bw, db, Asb]"""
    bc, dc, Asc, bw, db, Asb = vars_array

    penalty = 0.0

    # 1) Geometric constraints:
    # column width should not exceed depth: bc <= dc  (Equation 10)
    if bc - dc > 0:
        penalty += (bc - dc) * PENALTY_COEFF

    # beam depth proportion: 1.5 * bw <= db <= 2.0 * bw (Equations 19,20)
    if db < 1.5 * bw:
        penalty += (1.5 * bw - db) * PENALTY_COEFF
    if db > 2.0 * bw:
        penalty += (db - 2.0 * bw) * PENALTY_COEFF

    # compatibility constraint: column width >= beam width (Equation 40)
    if bw - bc > 0:
        penalty += (bw - bc) * PENALTY_COEFF

    # 2) Size limits (side constraints)
    for v, low, up in zip([bc, dc, Asc, bw, db, Asb], lb, ub):
        if v < low:
            penalty += (low - v) * PENALTY_COEFF
        if v > up:
            penalty += (v - up) * PENALTY_COEFF

    # 3) Steel ratio constraints (columns & beams)
    # Column steel ratio rho_c = Asc / (bc * dc)
    Agc = bc * dc
    rho_c = Asc / Agc if Agc > 0 else 1.0
    # ACI: ρmin = 0.01 (1%) and ρmax = 0.08 (8%)
    if rho_c < 0.01:
        penalty += (0.01 - rho_c) * PENALTY_COEFF
    if rho_c > 0.08:
        penalty += (rho_c - 0.08) * PENALTY_COEFF

    # Beam steel ratio rho_b = Asb / (bw * db)
    Agb = bw * db
    rho_b = Asb / Agb if Agb > 0 else 1.0
    # apply approximate beam limits: use ρmin and an upper limit related to ductility (paper mentions 75% of balanced)
    rho_b_min = max(0.001, 0.25 * (math.sqrt(f_c) / f_y) * 1e-3)  # approximate ρmin; safe lower
    if rho_b < rho_b_min:
        penalty += (rho_b_min - rho_b) * PENALTY_COEFF

    # upper bound: approximate ρmax_b = 0.08 as safe cap
    if rho_b > 0.08:
        penalty += (rho_b - 0.08) * PENALTY_COEFF

    # 4) Flexural capacity (beam) simplified check (Equation 21..25)
    # compute nominal moment capacity Mn (approx) and check 0.9 * Mn >= Mu
    # NOTE: The exact Mu should come from frame analysis; we approximate a required Mu from a sample applied load
    # For the optimizer we consider a sample required Mu_per_beam (this is a simplification)
    Mu_required = 100e3  # Nm per critical beam location (user should replace with actual factored moment)
    # compute a using ACI-style depth parameter:
    # a = (As * f_y) / (0.85 * f_c * bw)
    if bw > 0 and f_c > 0:
        a = (Asb * f_y) / (0.85 * f_c * bw)
        z = db - a / 2.0
        Mn = Asb * f_y * z  # approximate Nominal moment capacity (N-m)
        if 0.9 * Mn < Mu_required:
            # penalize proportionally to the shortfall
            shortfall = (Mu_required - 0.9 * Mn) / (Mu_required + 1e-9)
            penalty += shortfall * PENALTY_COEFF

    # 5) Column axial capacity simplified (Equation 11..13)
    # approximate Pu_required per column
    Pu_required = 500e3  # N (example) - user should set as actual factored axial loads
    # approximate Pn = 0.85 * f_c * Agc + Asc * f_y
    Pn = 0.85 * f_c * Agc + Asc * f_y
    if phi_column * Pn < Pu_required:
        short = (Pu_required - phi_column * Pn) / (Pu_required + 1e-9)
        penalty += short * PENALTY_COEFF

    # 6) Shear constraint (approx): Vn >= Vu
    # Use a sample Vu and approximate Vc + Vs
    Vu_required = 100e3  # N
    # concrete shear contribution Vc approx = 0.17 * math.sqrt(f_c) * bw * db
    Vc = 0.17 * math.sqrt(f_c) * bw * db
    # assume minimal stirrups -> Vs ~ 0.0; so require Vc >= Vu else penalize
    if Vc < Vu_required:
        penalty += (Vu_required - Vc) * PENALTY_COEFF

    # 7) Crack width & minimum beam width constraint (approx)
    # ensure bw allows bar placement -> bw >= br_min (approx)
    br_min = 0.025  # 25 mm approx min width for placement (example)
    if bw < br_min:
        penalty += (br_min - bw) * PENALTY_COEFF

    # ------------------------------
    # Compute objective cost: concrete + steel + formwork
    # ------------------------------
    # compute volumes using macros above
    Vcol = volume_column(bc, dc)
    Vbeam = volume_beam(bw, db)
    Lcbars = approx_column_bar_length()
    Lbbars = approx_beam_bar_length()

    Vsteel_col = volume_steel_columns(Asc, Lcbars)
    Vsteel_beam = volume_steel_beams(Asb, Lbbars)

    # steel masses (kg)
    mass_steel_col = Vsteel_col * rho_steel
    mass_steel_beam = Vsteel_beam * rho_steel
    total_steel_mass = mass_steel_col + mass_steel_beam

    # formwork areas
    Aform_col = area_formwork_columns(bc, dc)
    Aform_beam = area_formwork_beams(bw, db)

    # costs
    cost_concrete = (Vcol + Vbeam) * cost_concrete_per_m3
    cost_steel = total_steel_mass * cost_steel_per_kg
    cost_formwork = (Aform_col + Aform_beam) * cost_formwork_per_m2

    total_cost = cost_concrete + cost_steel + cost_formwork

    # add penalties
    fitness = total_cost + penalty

    return float(fitness), float(total_cost), float(penalty)

# Wrap for picklability in Spark map
def evaluate_individual(position):
    """ position: list or 1D-numpy """
    fitness, raw_cost, penalty = rc_frame_cost_and_penalty(position)
    return (position.tolist() if isinstance(position, np.ndarray) else list(position), fitness, raw_cost, penalty)

# ------------------------------
# Rao-1 + FISA hybrid optimizer implementation (population on driver, fitness eval on Spark)
# ------------------------------
def rao1_fisa_pyspark(pop_size=POP_SIZE, dim=DIM, lb_arr=lb, ub_arr=ub, max_iter=MAX_ITER):
    # initialize population
    Positions = np.random.rand(pop_size, dim) * (ub_arr - lb_arr) + lb_arr
    # evaluate initial population with Spark
    rdd = sc.parallelize(Positions.tolist(), numSlices=min(pop_size, 8))
    evals = rdd.map(evaluate_individual).collect()
    fitness_vals = np.array([e[1] for e in evals])
    raw_costs = np.array([e[2] for e in evals])
    penalties = np.array([e[3] for e in evals])

    history = []
    best_overall = None
    start_time = time.time()

    for k in range(max_iter):
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

        # Evaluate population in parallel with Spark
        rdd = sc.parallelize(Positions.tolist(), numSlices=min(pop_size, 8))
        evals = rdd.map(evaluate_individual).collect()
        new_fitness = np.array([e[1] for e in evals])
        new_raw_costs = np.array([e[2] for e in evals])
        new_penalties = np.array([e[3] for e in evals])

        # greedy replacement: accept if better
        for i in range(pop_size):
            if new_fitness[i] <= fitness_vals[i]:
                fitness_vals[i] = new_fitness[i]
                raw_costs[i] = new_raw_costs[i]
                penalties[i] = new_penalties[i]
            else:
                # revert to old position
                Positions[i] = Positioncopy[i].copy()

        best_idx = int(np.argmin(fitness_vals))
        best_score = float(fitness_vals[best_idx])
        best_solution = Positions[best_idx].copy()
        best_raw_cost = float(raw_costs[best_idx])
        best_penalty = float(penalties[best_idx])

        # store history periodically
        if (k + 1) % 100 == 0 or k == 0:
            elapsed = time.time() - start_time
            print(f"Iter {(k+1):5d}/{max_iter:5d} | Best fitness: {best_score:.4f} | Raw cost: {best_raw_cost:.4f} | Penalty: {best_penalty:.4f} | elapsed: {elapsed:.1f}s")
            history.append({
                "iter": k+1,
                "best_fitness": best_score,
                "best_raw_cost": best_raw_cost,
                "best_penalty": best_penalty,
                "x": best_solution.tolist()
            })

        # track global best
        if best_overall is None or best_score < best_overall["best_fitness"]:
            best_overall = {
                "best_fitness": best_score,
                "best_raw_cost": best_raw_cost,
                "best_penalty": best_penalty,
                "x": best_solution.copy(),
                "iter": k+1
            }

    # Save results to CSVs
    results_df = pd.DataFrame(history)
    out_dir = os.getcwd()
    results_csv = os.path.join(out_dir, "rao1_fisa_rc_frame_history.csv")
    best_csv = os.path.join(out_dir, "rao1_fisa_rc_frame_best.csv")
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

    print("Optimization complete.")
    print("History saved to:", results_csv)
    print("Best solution saved to:", best_csv)
    return best_overall

# ------------------------------
# Run when invoked
# ------------------------------
if __name__ == "__main__":
    try:
        best = rao1_fisa_pyspark()
        print("Final best:", best)
    finally:
        sc.stop()
