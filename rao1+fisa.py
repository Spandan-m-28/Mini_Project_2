# rao1_fisa_pyspark.py
# PySpark implementation of Rao-1 + FISA style update loop.
# - Includes the pressure_vessel objective from your original script for quick test
# - Shows clear placeholder where to implement frame cost from your attached PDF (manuscript_ijocta03259.pdf).
#
# Usage:
#   spark-submit rao1_fisa_pyspark.py
#
# Notes:
# - This implementation broadcasts the full population each iteration (ok for modest pop_size).
# - The objective can be replaced by a more complex structural-cost evaluator. See placeholder below.
# - Outputs results/tracking CSV to the output path configured below.

from pyspark.sql import SparkSession
import numpy as np
import random
import math
import json
import os
from datetime import datetime

# --- Configuration ---
POP_SIZE = 200          # example pop size (tune for your cluster)
DIM = 4                 # dimension (for pressure vessel example). For frame optimization set accordingly
MAX_FES = 30000         # maximum function evaluations (global)
LB = np.array([0.0625, 0.0625, 10.0, 10.0])   # lower bounds (pressure vessel)
UB = np.array([99.0, 99.0, 200.0, 240.0])    # upper bounds (pressure vessel)
TRACK_EVERY_FE = 1000   # save tracking results every this many function evaluations
OUTPUT_DIR = "rao1_fisa_results"  # local or HDFS path (make sure cluster user can write)
SEED = 42

# For reproducibility
random.seed(SEED)
np.random.seed(SEED)

# --- Create Spark session ---
spark = SparkSession.builder.appName("Rao1_FISA_PySpark").getOrCreate()
sc = spark.sparkContext

# Ensure output directory exists (driver-local; change for HDFS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Objective functions ---
def pressure_vessel(solution):
    """
    Direct translation of your original pressure_vessel function.
    Returns very large penalty for constraint violation.
    """
    x1, x2, x3, x4 = solution
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = 1296000 - (4.0 / 3.0) * math.pi * (x3 ** 3) - math.pi * (x3 ** 2) * x4
    g4 = x4 - 240.0

    if (g1 <= 0) and (g2 <= 0) and (g3 <= 0) and (g4 <= 0):
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * (x3 ** 2) + 3.1661 * (x1 ** 2) * x4 + 19.84 * (x1 ** 2) * x3
    else:
        return 1e10

# -------------------------
# PLACEHOLDER for your real objective:
# The manuscript_ijocta03259.pdf contains a structural cost function:
#   Minimize total cost = sum(cost_concrete + cost_steel + cost_formwork) (Eq. (1) and (3) in the PDF).
# Implement an evaluator similar to the pressure_vessel() above but computing:
#   - decode individual's chromosome -> member sizes, reinforcement indices (see Section 4 of the PDF)
#   - run structural analysis (or use simplified formulas) to produce internal actions
#   - compute cost using unit rates (Table 2 in the PDF) and add penalties for constraint violations
# Example skeleton function:
def evaluate_individual_frame(individual, params=None):
    """
    Replace this stub with the cost/evaluation logic from manuscript_ijocta03259.pdf.
    - `individual` is a numpy array or Python list encoding decision variables (widths, heights, reinforcement indices...).
    - `params` can carry unit costs, topology, load cases, code parameters.
    Return: float fitness (lower is better).
    """
    # Example (VERY simplified): decode just two members' area and compute dummy cost.
    # **YOU MUST REPLACE THIS** with the real decode/evaluate steps from the paper.
    # Use params to broadcast any large structures (topology, unit rates).
    a = float(individual[0])
    b = float(individual[1]) if len(individual) > 1 else a
    # simplified "cost"
    cost_concrete = 112.13 * (a * b * 1.0)   # unit cost * volume (dummy)
    cost_steel = 1.30 * (a * 10.0)           # dummy
    penalty = 0.0
    return cost_concrete + cost_steel + penalty

# -------------------------

# Choose which objective to use:
USE_FRAME_OBJECTIVE = False   # set True to use evaluate_individual_frame (after you implement it)
def evaluate(solution):
    if USE_FRAME_OBJECTIVE:
        return evaluate_individual_frame(solution, params=None)
    else:
        return pressure_vessel(solution)

# --- Utility functions for PySpark mapping ---
def make_initial_population(pop_size, dim, lb, ub):
    # returns a numpy array shape (pop_size, dim)
    return np.random.rand(pop_size, dim) * (ub - lb) + lb

# RDD element structure: (idx, position_list, fitness)
def pack_population(pop_array):
    pop = []
    for i in range(pop_array.shape[0]):
        pos = pop_array[i].tolist()
        fitness = float(evaluate(pos))
        pop.append( (i, pos, fitness) )
    return pop

def update_individual(args):
    """
    Map function to update a single individual using Rao-1 + FISA style step.
    It receives a tuple:
       (i, pos, fitness, pop_broadcast_value, best_pos, worst_pos, lb, ub)
    and returns (i, new_pos, new_fitness)
    """
    (i, pos, fitness, pop_list, best_pos, worst_pos, lb_list, ub_list) = args
    dim = len(pos)
    pos_arr = np.array(pos, dtype=float)
    pos_copy = pos_arr.copy()
    pop_arr = np.array(pop_list)  # shape (pop_size, dim)

    # apply Rao-1 + FISA-like update per-dimension
    for j in range(dim):
        r1 = random.random()
        r2 = random.random()
        term1 = r1 * (best_pos[j] - worst_pos[j])

        # pick random other index
        rand_idx = random.randint(0, pop_arr.shape[0] - 1)
        mx = max(pos_copy[j], pop_arr[rand_idx][j])
        term2 = r2 * (pos_copy[j] - mx)

        pos_arr[j] += term1 + term2

        # clip to bounds
        pos_arr[j] = max(lb_list[j], min(ub_list[j], pos_arr[j]))

    new_fit = evaluate(pos_arr.tolist())
    # If new is worse (higher), revert to copy (as in your original code)
    if new_fit > fitness:
        return (i, pos_copy.tolist(), float(fitness))
    else:
        return (i, pos_arr.tolist(), float(new_fit))


# --- Main optimization loop (driver-driven broadcast) ---
def run_rao1_fisa(sc,
                   pop_size=POP_SIZE,
                   dim=DIM,
                   lb=LB,
                   ub=UB,
                   max_fes=MAX_FES,
                   track_every=TRACK_EVERY_FE):
    # initialize population on driver
    pop = make_initial_population(pop_size, dim, lb, ub)
    # pack into RDD items with fitness
    pop_items = pack_population(pop)
    # Create an RDD (we will re-broadcast/populate each iteration)
    pop_rdd = sc.parallelize(pop_items, numSlices=min(8, pop_size))

    fes = 0
    iter_no = 0
    results_tracking = []  # list of dicts for later saving

    max_iter = max_fes // pop_size
    if max_iter < 1:
        max_iter = 1

    while fes < max_fes and iter_no < max_iter:
        # collect current population to driver for easy best/worst identification
        pop_list = pop_rdd.collect()  # relatively small (pop_size)
        # pop_list elements: (i, pos, fitness)
        positions = np.array([item[1] for item in pop_list])
        fitnesses = np.array([item[2] for item in pop_list])

        best_idx = int(np.argmin(fitnesses))
        worst_idx = int(np.argmax(fitnesses))
        best_pos = positions[best_idx].tolist()
        worst_pos = positions[worst_idx].tolist()
        best_score = float(fitnesses[best_idx])

        # tracking
        if (fes % track_every) == 0:
            results_tracking.append({
                "fes": fes,
                "iter": iter_no,
                "best_score": best_score,
                "best_pos": best_pos
            })
            print(f"[Iter {iter_no}] FEs={fes} Best={best_score:.6f}")

        # broadcast current population and bounds and best/worst
        b_pop = sc.broadcast(positions.tolist())
        b_best = sc.broadcast(best_pos)
        b_worst = sc.broadcast(worst_pos)
        b_lb = sc.broadcast(lb.tolist())
        b_ub = sc.broadcast(ub.tolist())

        # Prepare RDD elements for update: convert to tuples with broadcast data included
        # To avoid re-collect inside map, create an RDD of tuples where each partition receives needed broadcasts.
        update_input = pop_rdd.map(lambda item: (
            item[0],    # idx
            item[1],    # pos
            item[2],    # fitness
            b_pop.value,
            b_best.value,
            b_worst.value,
            b_lb.value,
            b_ub.value
        ))

        # Perform update in parallel
        updated = update_input.map(update_individual)

        # force evaluation and create new pop_rdd
        updated_list = updated.collect()
        fes += pop_size
        iter_no += 1

        # make new RDD for next iteration
        pop_rdd = sc.parallelize(updated_list, numSlices=min(8, pop_size))

        # optional: unpersist broadcasts
        b_pop.unpersist()
        b_best.unpersist()
        b_worst.unpersist()
        b_lb.unpersist()
        b_ub.unpersist()

    # final collection
    final_pop = pop_rdd.collect()
    # compute final best
    fitnesses = np.array([item[2] for item in final_pop])
    positions = np.array([item[1] for item in final_pop])
    best_idx = int(np.argmin(fitnesses))
    best_score = float(fitnesses[best_idx])
    best_pos = positions[best_idx].tolist()

    # Save tracking results as CSV-like file
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(OUTPUT_DIR, f"rao1_fisa_tracking_{ts}.csv")
    with open(out_path, "w") as fh:
        fh.write("fes,iter,best_score,best_pos\n")
        for rec in results_tracking:
            fh.write(f"{rec['fes']},{rec['iter']},{rec['best_score']},\"{json.dumps(rec['best_pos'])}\"\n")
    print("Saved tracking to", out_path)

    # Also save final population
    pop_out_path = os.path.join(OUTPUT_DIR, f"rao1_fisa_population_{ts}.csv")
    with open(pop_out_path, "w") as fh:
        fh.write("idx,fitness,position\n")
        for item in final_pop:
            fh.write(f"{item[0]},{item[2]},\"{json.dumps(item[1])}\"\n")
    print("Saved final population to", pop_out_path)

    return {
        "best_score": best_score,
        "best_pos": best_pos,
        "tracking_file": out_path,
        "population_file": pop_out_path
    }

if __name__ == "__main__":
    print("Starting Rao-1 + FISA PySpark run")
    res = run_rao1_fisa(sc,
                        pop_size=POP_SIZE,
                        dim=DIM,
                        lb=LB,
                        ub=UB,
                        max_fes=MAX_FES,
                        track_every=TRACK_EVERY_FE)
    print("Run complete. Best score:", res["best_score"])
    print("Best pos:", res["best_pos"])
    spark.stop()
