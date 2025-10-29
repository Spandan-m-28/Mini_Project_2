#!/usr/bin/env python3
"""
Rao1-like hybrid optimization for RC Frame cost optimization
Parallel fitness evaluation using PySpark.

Save as: rao_rcframe_pyspark.py
Run with: spark-submit rao_rcframe_pyspark.py --pop 80 --iters 200
"""

import argparse
import math
import numpy as np
import random
import csv
import os
from pyspark.sql import SparkSession
from pyspark import SparkConf

# -----------------------------
# Problem configuration (editable)
# -----------------------------
# Simple regular frame geometry: n_stories x n_bays
N_STORIES = 5
N_BAYS = 2
SPAN = 6.0              # span length (m) for beams
STOREY_HEIGHT = 3.0     # (m)
GRAVITY_LOAD = 35.0     # kN/m applied at each floor (from paper example)
CONCRETE_DENSITY = 2400 # kg/m3
STEEL_DENSITY = 7850    # kg/m3
Fyd = 500.0             # steel yield (MPa) (typical)
fcd = 25.0              # concrete design compressive strength (MPa) (approx)

# Cost unit rates (adjust as needed)
C_CONCRETE = 112.13     # €/m3 (from paper table - just an example)
C_STEEL = 1.30          # €/kg
C_FORMWORK = 25.05      # €/m2 (beams) approximate

# Discrete options (value-encoding arrays)
WIDTHS = np.arange(0.20, 0.80+1e-9, 0.05)   # m (200mm to 800mm)
HEIGHTS = np.arange(0.20, 1.20+1e-9, 0.05)  # m (200mm to 1200mm)
BAR_DIAMETERS = np.array([8,10,12,16,20,25,32])  # mm
STIRRUP_SPACING = np.array([0.10,0.125,0.15,0.20,0.25,0.30,0.35])  # m

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Chromosome structure
# -----------------------------
# We'll encode:
# - Each beam: width_idx, height_idx, top_bar_idx, bottom_bar_idx, stirrup_spacing_idx
# - Each column: width_idx, height_idx, long_bar_idx, stirrup_spacing_idx
# For simplicity, assume same encoding per beam/column type and group symmetric members.

NB_BEAMS = N_STORIES * N_BAYS * 1   # beams per floor per bay (one beam per span)
NB_COLUMNS = (N_BAYS + 1) * (N_STORIES + 1)  # approximate nodes -> columns (simplified)

# To reduce chromosome size, group identical members by story/bay symmetry (simple grouping)
# We'll use a small grouping: one unique beam per floor (assuming identical across bays), one column per column line.
GROUPED_BEAMS = N_STORIES  # one beam size per storey (applied to all bays)
GROUPED_COLUMNS = N_BAYS + 1  # one column size per column line (all stories same)

# Chromosome length calculation
BEAM_GENE_LEN = 5      # width_idx, height_idx, top_bar_idx, bottom_bar_idx, stirrup_idx
COLUMN_GENE_LEN = 4    # width_idx, height_idx, long_bar_idx, stirrup_idx
CHROMOSOME_LEN = GROUPED_BEAMS * BEAM_GENE_LEN + GROUPED_COLUMNS * COLUMN_GENE_LEN

# helper indices offsets
BEAM_OFFSET = 0
COLUMN_OFFSET = GROUPED_BEAMS * BEAM_GENE_LEN

# -----------------------------
# Helper decode/encode functions
# -----------------------------
def random_chromosome():
    """Create a random chromosome (value-encoding indices)."""
    chrom = []
    # beams
    for _ in range(GROUPED_BEAMS):
        chrom.append(random.randrange(len(WIDTHS)))   # width_idx
        chrom.append(random.randrange(len(HEIGHTS)))  # height_idx
        chrom.append(random.randrange(len(BAR_DIAMETERS)))  # top bars
        chrom.append(random.randrange(len(BAR_DIAMETERS)))  # bottom bars
        chrom.append(random.randrange(len(STIRRUP_SPACING))) # stirrup spacing
    # columns
    for _ in range(GROUPED_COLUMNS):
        chrom.append(random.randrange(len(WIDTHS)))
        chrom.append(random.randrange(len(HEIGHTS)))
        chrom.append(random.randrange(len(BAR_DIAMETERS)))  # long bars
        chrom.append(random.randrange(len(STIRRUP_SPACING)))
    return np.array(chrom, dtype=int)

def decode_chromosome(chrom):
    """Return structured properties for beams and columns."""
    beams = []
    cols = []
    idx = 0
    for b in range(GROUPED_BEAMS):
        w = WIDTHS[chrom[idx]]; h = HEIGHTS[chrom[idx+1]]
        top_bar = BAR_DIAMETERS[chrom[idx+2]]
        bot_bar = BAR_DIAMETERS[chrom[idx+3]]
        stir = STIRRUP_SPACING[chrom[idx+4]]
        beams.append({'b': w, 'h': h, 'top_bar': top_bar, 'bot_bar': bot_bar, 'stirrup': stir})
        idx += BEAM_GENE_LEN
    for c in range(GROUPED_COLUMNS):
        w = WIDTHS[chrom[idx]]; h = HEIGHTS[chrom[idx+1]]
        long_bar = BAR_DIAMETERS[chrom[idx+2]]
        stir = STIRRUP_SPACING[chrom[idx+3]]
        cols.append({'b': w, 'h': h, 'long_bar': long_bar, 'stirrup': stir})
        idx += COLUMN_GENE_LEN
    return beams, cols

# -----------------------------
# Surrogate structural evaluation
# -----------------------------
def area_of_bar(d_mm):
    """Area of one bar (m^2)."""
    return (math.pi * (d_mm/1000.0)**2) / 4.0

def evaluate_structure(chrom):
    """
    Evaluate cost + penalty for a chromosome.
    NOTE: This is a surrogate evaluator — replace with FE analysis for production.
    """
    beams, cols = decode_chromosome(chrom)
    # 1) approximate internal actions
    # Beam moment (uniformly distributed floor load): w per unit length = GRAVITY_LOAD (kN/m)
    # For simply supported beam of span L: M = w*L^2/8 (kN*m)
    # Column axial = tributary loads from floors above (simplified).
    # We'll compute costs and simple capacities.
    total_concrete_vol = 0.0  # m3
    total_steel_weight = 0.0  # kg
    penalty = 0.0

    # beam contributions (per bay)
    for story in range(N_STORIES):
        beam_spec = beams[min(story, len(beams)-1)]
        b = beam_spec['b']; h = beam_spec['h']
        # beam volume (one beam across one bay) : b*h*L
        vol_beam = b * h * SPAN
        total_concrete_vol += vol_beam * N_BAYS  # beams across all bays
        # steel: approximate top+bottom two bars each (assume 2 bars each side)
        top_area = 2 * area_of_bar(beam_spec['top_bar'])
        bot_area = 2 * area_of_bar(beam_spec['bot_bar'])
        steel_area = top_area + bot_area  # m2
        # steel volume = steel_area * length
        steel_vol = steel_area * SPAN
        steel_mass = steel_vol * STEEL_DENSITY
        total_steel_weight += steel_mass * 1.0 * 1.0 * 1.0  # per beam per bay
        # formwork area
        # we'll add a simple shear/moment check
        w = GRAVITY_LOAD  # kN/m
        M_beam = w * SPAN**2 / 8.0  # kN*m
        # approximate bending capacity: M_rd = z * As * fyd (z ~ 0.9d)
        d = h - 0.05  # effective depth ~ h - cover (0.05m)
        z = 0.9 * d
        As_provided = steel_area * 1.0  # m2
        # convert units: fyd (MPa) -> N/mm2 ; use consistent units
        # M_rd (kN*m) = As(m2) * fyd(MPa) * z(m) * (1e3)  [since MPa = N/mm2]
        M_rd = As_provided * Fyd * z * 1e-3  # approx in kN*m (rough conversion)
        if M_rd < M_beam:
            # penalty proportional to shortfall
            short = (M_beam - M_rd) / (M_beam + 1e-9)
            penalty += 10.0 * short  # penalty scale

    # column contributions
    # assume columns run full height and tributary floor loads sum into axial
    # Count columns per column line
    for c_idx in range(GROUPED_COLUMNS):
        col_spec = cols[min(c_idx, len(cols)-1)]
        b = col_spec['b']; h = col_spec['h']
        vol_col = b * h * (STOREY_HEIGHT * N_STORIES)
        total_concrete_vol += vol_col
        # long bars: assume 4 bars per column
        long_bar_area = 4 * area_of_bar(col_spec['long_bar'])
        steel_vol = long_bar_area * STOREY_HEIGHT * N_STORIES
        steel_mass = steel_vol * STEEL_DENSITY
        total_steel_weight += steel_mass
        # axial load: tributary floors (simplified)
        # each column supports floor loads of its tributary area. We'll approximate axial demand:
        tributary_area = SPAN * (SPAN / (N_BAYS + 1))  # rough
        axial = GRAVITY_LOAD * tributary_area * N_STORIES  # kN total
        # column capacity: N_rd = A_c * fcd * gamma (very simplified)
        A_c = b * h  # m2
        # fcd in MPa -> kN/m2 multiply by 1e3
        N_rd = A_c * fcd * 1e3  # kN
        if N_rd < axial:
            short = (axial - N_rd) / (axial + 1e-9)
            penalty += 20.0 * short

    # compute costs
    cost_concrete = total_concrete_vol * C_CONCRETE
    cost_steel = total_steel_weight * C_STEEL / 1.0  # steel mass is in kg
    # formwork approx: beam formwork = beam perimeter * length * bays
    beam_form_area = 0.0
    for story in range(N_STORIES):
        beam_spec = beams[min(story, len(beams)-1)]
        # assume formwork area per beam ~ 2*(b+h)*L
        beam_form_area += 2 * (beam_spec['b'] + beam_spec['h']) * SPAN * N_BAYS
    cost_formwork = beam_form_area * C_FORMWORK

    total_cost = cost_concrete + cost_steel + cost_formwork
    # construct fitness as cost * (1 + penalty)
    fitness = total_cost * (1.0 + penalty)
    # Return large penalty if any dimension violates min criteria (b/h ratio)
    # min width = 0.2 m, b/h >= 0.5
    for beam in beams:
        if beam['b'] < 0.20 or beam['b'] / beam['h'] < 0.5:
            fitness += 1e6
    for col in cols:
        if col['b'] < 0.20 or col['b'] / col['h'] < 0.5:
            fitness += 1e6

    return float(fitness), float(total_cost), float(penalty)

# -----------------------------
# Parallel evaluation wrapper for PySpark
# -----------------------------
def spark_evaluate_population(spark, population):
    """
    population: numpy array of shape (pop_size, chrom_len)
    returns: fitness array, cost array, penalty array
    """
    sc = spark.sparkContext
    # Broadcast constants / lookups if needed (we use global arrays; small)
    # Convert population to list for RDD
    pop_list = [chrom.tolist() for chrom in population]
    rdd = sc.parallelize(pop_list, numSlices=min(len(pop_list), sc.defaultParallelism * 2))
    # map evaluation
    def map_eval(chrom_list):
        chrom = np.array(chrom_list, dtype=int)
        fitness, cost, penalty = evaluate_structure(chrom)
        return (chrom_list, fitness, cost, penalty)
    results = rdd.map(map_eval).collect()
    # reconstruct arrays
    fitness_vals = np.zeros(len(results))
    cost_vals = np.zeros(len(results))
    penalty_vals = np.zeros(len(results))
    pop_out = []
    for i, (chrom_list, fit, cost, pen) in enumerate(results):
        fitness_vals[i] = fit
        cost_vals[i] = cost
        penalty_vals[i] = pen
        pop_out.append(np.array(chrom_list, dtype=int))
    pop_out = np.vstack(pop_out)
    return pop_out, fitness_vals, cost_vals, penalty_vals

# -----------------------------
# Rao1-like optimization loop (driver)
# -----------------------------
def rao1_pyspark(pop_size=60, max_iter=100, checkpoint_every=10, out_prefix="results_tracking"):
    # init Spark
    spark = SparkSession.builder.appName("Rao1_RCFrame_Optimization").getOrCreate()
    # initialize population
    population = np.vstack([random_chromosome() for _ in range(pop_size)])
    # initial evaluation (parallel)
    population, fitness_vals, cost_vals, penalty_vals = spark_evaluate_population(spark, population)

    best_history = []
    best_overall = None
    best_fit_overall = np.inf

    for it in range(max_iter):
        # find best and worst
        best_idx = np.argmin(fitness_vals)
        worst_idx = np.argmax(fitness_vals)
        best_pos = population[best_idx].copy()
        worst_pos = population[worst_idx].copy()

        pop_copy = population.copy()

        # Rao1-like position update (discrete indices)
        for i in range(pop_size):
            for j in range(CHROMOSOME_LEN):
                r1 = random.random()
                r2 = random.random()
                term1 = r1 * (best_pos[j] - worst_pos[j])
                # second term uses a random partner
                rand_idx = random.randint(0, pop_size - 1)
                MX = max(pop_copy[i][j], pop_copy[rand_idx][j])
                term2 = r2 * (pop_copy[i][j] - MX)
                # update in continuous index-space
                new_val = population[i][j] + int(round(term1 + term2))
                # clip to legal index range per gene
                # gene ranges depend on position in chromosome
                # compute gene-specific max index
                if j < COLUMN_OFFSET:
                    # beam gene
                    local_gene_pos = j % BEAM_GENE_LEN
                    if local_gene_pos == 0:
                        max_idx = len(WIDTHS)-1
                    elif local_gene_pos == 1:
                        max_idx = len(HEIGHTS)-1
                    elif local_gene_pos in (2,3):
                        max_idx = len(BAR_DIAMETERS)-1
                    else:
                        max_idx = len(STIRRUP_SPACING)-1
                else:
                    # column gene
                    local_gene_pos = (j - COLUMN_OFFSET) % COLUMN_GENE_LEN
                    if local_gene_pos == 0:
                        max_idx = len(WIDTHS)-1
                    elif local_gene_pos == 1:
                        max_idx = len(HEIGHTS)-1
                    elif local_gene_pos == 2:
                        max_idx = len(BAR_DIAMETERS)-1
                    else:
                        max_idx = len(STIRRUP_SPACING)-1
                new_val = max(0, min(max_idx, new_val))
                population[i][j] = int(new_val)

        # Evaluate newly generated population in parallel
        population, new_fitness_vals, new_cost_vals, new_penalty_vals = spark_evaluate_population(spark, population)

        # greedy replacement: keep new individuals if better
        for i in range(pop_size):
            if new_fitness_vals[i] < fitness_vals[i]:
                fitness_vals[i] = new_fitness_vals[i]
                cost_vals[i] = new_cost_vals[i]
                penalty_vals[i] = new_penalty_vals[i]
            else:
                # revert individual to previous pop_copy
                population[i] = pop_copy[i]

        # track best
        best_idx = np.argmin(fitness_vals)
        best_fit = fitness_vals[best_idx]
        best_chrom = population[best_idx].copy()

        if best_fit < best_fit_overall:
            best_fit_overall = best_fit
            best_overall = best_chrom.copy()

        # checkpointing & logging
        if (it + 1) % checkpoint_every == 0 or it == 0:
            decoded_beams, decoded_cols = decode_chromosome(best_chrom)
            best_history.append((it+1, best_fit, cost_vals[best_idx], penalty_vals[best_idx], best_chrom.tolist()))
            print(f"Iter {it+1}/{max_iter} | Best fitness: {best_fit:.3f} | Cost: {cost_vals[best_idx]:.2f} | Penalty: {penalty_vals[best_idx]:.3f}")

    # Save results
    out_csv = out_prefix + ".csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iter", "BestFitness", "Cost", "Penalty", "Chromosome"])
        for rec in best_history:
            writer.writerow(rec)

    # Save best chromosome numpy
    np.save("best_solution.npy", best_overall)

    print("Optimization complete. Best fitness:", best_fit_overall)
    spark.stop()
    return best_overall, best_fit_overall

# -----------------------------
# Main / argument parsing
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", type=int, default=60, help="Population size")
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    parser.add_argument("--checkpoint", type=int, default=10, help="checkpoint every N iterations")
    args = parser.parse_args()

    best_chrom, best_fit = rao1_pyspark(pop_size=args.pop, max_iter=args.iters, checkpoint_every=args.checkpoint)
    print("Best chromosome saved to best_solution.npy")
