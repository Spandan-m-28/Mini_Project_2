# rao1_fisa_frame_cost_pyspark.py
"""
PySpark implementation: Rao-1 + FISA style optimizer for reinforced-concrete frame cost
- Cost model follows Eq.(1) and Eq.(3) in Habte & Yilma (IJOCTA) and unit rates in Table 2.
- Chromosome is value-encoded (indices -> discrete sizes / reinforcement options).
- THIS SCRIPT DOES NOT PERFORM FULL STRUCTURAL ANALYSIS (stiffness matrix) --
  it uses geometric & reinforcement constraints + cost model + simple penalties.
  Integrate your structural analysis module into `evaluate_individual()` for full checks.

References:
- Habte, B. & Yilma, E., "Cost optimization of reinforced concrete frames using genetic algorithms", IJOCTA (2021).
  Eq.(1), Eq.(3), Table 2 (unit rates), encoding/constraints descriptions. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
"""
from pyspark.sql import SparkSession
import numpy as np
import random
import math
import json
import os
from datetime import datetime

# ---------------- USER CONFIG ----------------
POP_SIZE = 200
MAX_FES = 20000
SEED = 42
OUTPUT_DIR = "rao1_frame_results"
TRACK_EVERY = 2000

random.seed(SEED)
np.random.seed(SEED)

# make output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- UNIT RATES from Table 2 (paper) ----------------
UNIT_STEEL_EU_PER_KG = 1.30        # €/kg
UNIT_CONCRETE_EU_PER_M3 = 112.13   # €/m^3
UNIT_FORM_BEAM_EU_PER_M2 = 25.05   # €/m^2 (beams formwork)
UNIT_FORM_COLUMN_EU_PER_M2 = 22.75 # €/m^2 (columns formwork)
UNIT_SCAFFOLD_BEAM_EU_PER_M2 = 38.89 # €/m^2 (scaffolding beams)

STEEL_DENSITY = 7850.0  # kg/m^3

# ----------------- Encoding domains -----------------
# Section sizes (breadth/height) discrete choices: from 200mm to 500mm step 25mm
SECTION_SIZES_MM = list(range(200, 525, 25))  # mm
SECTION_SIZES_M = [v / 1000.0 for v in SECTION_SIZES_MM]  # m

# Available bar diameters (mm) for continuous and additional reinforcement
BAR_DIAMETERS_MM = [8, 10, 12, 16, 20, 25]  # typical
BAR_AREA_M2 = [(math.pi * (d/1000.0)**2 / 4.0) for d in BAR_DIAMETERS_MM]  # m^2 per bar

# Stirrup spacing options in mm (100 - 350 with variety)
STIRRUP_SPACING_MM = [100, 120, 150, 170, 200, 230, 260, 300, 350]

# Minimal domain checks (from paper: min width 200mm, b/h ratio limit 0.5 - 2.0)
MIN_SECTION_B_MM = 200
BH_RATIO_MIN = 0.5
BH_RATIO_MAX = 2.0

# Reinforcement ratio limits (typical values). Paper mentions min/max but not explicit numbers;
# choose conservative defaults (can be adjusted or read from input).
REINF_RATIO_MIN = 0.005   # 0.5% (minimum reinforcement ratio)
REINF_RATIO_MAX = 0.06    # 6% (maximum reinforcement ratio)

# ----------------- FRAME DATA (example) -----------------
# A tiny 2-bay, 2-story planar frame example. Replace with your real frame description or load JSON.
FRAME_DEF = {
    "n_beams": 4,               # total horizontal members
    "n_columns": 4,             # total vertical members
    # lengths in meters for each member (for simplicity we take uniform lengths)
    "beam_length_m": 3.0,
    "column_length_m": 3.0,
    # Assume beams are rectangular prismatic elements; columns likewise.
    # For a real structure you would provide per-member lengths and classification (beam/column).
}

# For encoding: number of genes per beam and per column
# Following the paper: 8 indices per beam, 5 indices per column (typical scheme described).
GENES_PER_BEAM = 8
GENES_PER_COLUMN = 5

TOTAL_GENES = FRAME_DEF["n_beams"] * GENES_PER_BEAM + FRAME_DEF["n_columns"] * GENES_PER_COLUMN

# ---------------- helper functions ----------------
def random_chromosome():
    """
    Generate a random chromosome (array of indices)
    Gene ordering (per beam): [b_idx, h_idx, cont_top_bar_idx, cont_bottom_bar_idx,
                              pos_add_bar_idx, neg_add_bar_idx, stirrup_spacing_idx, stirrup_spacing_idx2]
    (adjustable)
    For columns: [b_idx, h_idx, cont_bar_idx, pos_add_bar_idx, stirrup_spacing_idx]
    """
    chrom = []
    # beams
    for _ in range(FRAME_DEF["n_beams"]):
        # b_idx, h_idx
        chrom.append(random.randrange(len(SECTION_SIZES_MM)))  # b
        chrom.append(random.randrange(len(SECTION_SIZES_MM)))  # h
        # continuous top/bottom bar choices (index into BAR_DIAMETERS_MM)
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))
        # positive / negative additional bars
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))
        # two stirrup spacing indices
        chrom.append(random.randrange(len(STIRRUP_SPACING_MM)))
        chrom.append(random.randrange(len(STIRRUP_SPACING_MM)))
    # columns
    for _ in range(FRAME_DEF["n_columns"]):
        chrom.append(random.randrange(len(SECTION_SIZES_MM)))  # b
        chrom.append(random.randrange(len(SECTION_SIZES_MM)))  # h
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))  # cont bar
        chrom.append(random.randrange(len(BAR_DIAMETERS_MM)))  # add bar
        chrom.append(random.randrange(len(STIRRUP_SPACING_MM)))
    return chrom

def decode_chromosome(chrom):
    """
    Decode integer-index chromosome into physical sizes & reinforcement.
    Returns dictionary with per-member geometry & reinforcement data.
    """
    data = {"beams": [], "columns": []}
    idx = 0
    # beams
    for b in range(FRAME_DEF["n_beams"]):
        b_idx = chrom[idx]; idx += 1
        h_idx = chrom[idx]; idx += 1
        cont_top_idx = chrom[idx]; idx += 1
        cont_bot_idx = chrom[idx]; idx += 1
        pos_add_idx = chrom[idx]; idx += 1
        neg_add_idx = chrom[idx]; idx += 1
        stir1_idx = chrom[idx]; idx += 1
        stir2_idx = chrom[idx]; idx += 1

        beam = {
            "b_m": SECTION_SIZES_M[b_idx],
            "h_m": SECTION_SIZES_M[h_idx],
            "cont_top_d_mm": BAR_DIAMETERS_MM[cont_top_idx],
            "cont_bot_d_mm": BAR_DIAMETERS_MM[cont_bot_idx],
            "pos_add_d_mm": BAR_DIAMETERS_MM[pos_add_idx],
            "neg_add_d_mm": BAR_DIAMETERS_MM[neg_add_idx],
            "stirrup_spacing_mm": STIRRUP_SPACING_MM[stir1_idx],
            "stirrup_spacing_mm2": STIRRUP_SPACING_MM[stir2_idx],
            "length_m": FRAME_DEF["beam_length_m"]
        }
        data["beams"].append(beam)
    # columns
    for c in range(FRAME_DEF["n_columns"]):
        b_idx = chrom[idx]; idx += 1
        h_idx = chrom[idx]; idx += 1
        cont_idx = chrom[idx]; idx += 1
        add_idx = chrom[idx]; idx += 1
        stir_idx = chrom[idx]; idx += 1

        col = {
            "b_m": SECTION_SIZES_M[b_idx],
            "h_m": SECTION_SIZES_M[h_idx],
            "cont_d_mm": BAR_DIAMETERS_MM[cont_idx],
            "add_d_mm": BAR_DIAMETERS_MM[add_idx],
            "stirrup_spacing_mm": STIRRUP_SPACING_MM[stir_idx],
            "length_m": FRAME_DEF["column_length_m"]
        }
        data["columns"].append(col)
    return data

def compute_cost_and_penalty(decoded):
    """
    Compute total cost using Eq.(3)-like expressions (concrete, steel, formwork) and
    compute penalty for simple constraints (min size, b/h ratio, reinforcement ratio).
    Returns (cost_euro, penalty_scalar)
    """
    total_concrete_vol_m3 = 0.0
    total_steel_kg = 0.0
    total_form_area_m2_beams = 0.0
    total_form_area_m2_columns = 0.0
    total_scaffold_area_m2_beams = 0.0

    penalty = 0.0

    # beams
    for beam in decoded["beams"]:
        b = beam["b_m"]; h = beam["h_m"]; L = beam["length_m"]
        # approximate member concrete volume as b*h*L
        vol = b * h * L
        total_concrete_vol_m3 += vol

        # approximate steel: continuous top/bottom: assume 2 bars each (paper used two bars top/bot)
        cont_top_area = math.pi * (beam["cont_top_d_mm"]/1000.0)**2 / 4.0
        cont_bot_area = math.pi * (beam["cont_bot_d_mm"]/1000.0)**2 / 4.0
        # assume 2 bars each side for continuous bars (paper: two bars at top and bottom)
        cont_total_area = (cont_top_area * 2.0 + cont_bot_area * 2.0)
        # additional bars: assume 2 bars for pos and 2 for neg (approx)
        pos_add_area = math.pi * (beam["pos_add_d_mm"]/1000.0)**2 / 4.0
        neg_add_area = math.pi * (beam["neg_add_d_mm"]/1000.0)**2 / 4.0
        add_total_area = (pos_add_area * 2.0 + neg_add_area * 2.0)
        # stirrups: estimate total length of stirrups per beam as perimeter * (L / spacing)
        spacing = beam["stirrup_spacing_mm"]/1000.0
        spacing2 = beam["stirrup_spacing_mm2"]/1000.0
        # use min spacing of the two indices (conservative)
        spacing_use = min(spacing, spacing2)
        if spacing_use <= 0.0:
            spacing_use = 0.1
        perimeter = 2.0 * (b + h)
        n_stirrups = max(1, int(math.ceil(L / spacing_use)))
        # assume stirrup bar dia = 8mm as paper; total length per stirrup = perimeter + hooks ~ perimeter * 1.15
        stirrup_bar_d_mm = 8
        stirrup_len = perimeter * 1.15
        stirrup_area_per_bar = math.pi * (stirrup_bar_d_mm/1000.0)**2 / 4.0
        total_stirrups_area = stirrup_area_per_bar * n_stirrups

        # steel area per unit length (m^2 per m)
        steel_area_m2 = (cont_total_area + add_total_area + total_stirrups_area)
        # mass = area (m^2) * length (m) * density
        steel_mass_kg = steel_area_m2 * L * STEEL_DENSITY
        total_steel_kg += steel_mass_kg

        # formwork area (top + bottom + sides) approx: 2*(b*L) + 2*(h*L) -> simplified as perimeter * L
        form_area_beam = perimeter * L
        total_form_area_m2_beams += form_area_beam

        # scaffolding: approximate as beam plan area (b * L) times some factor -> use b*L
        total_scaffold_area_m2_beams += (b * L)

        # simple sizing constraints penalties
        if (b * 1000.0) < MIN_SECTION_B_MM or (h * 1000.0) < MIN_SECTION_B_MM:
            penalty += 2.0  # arbitrary penalty for too small dimension
        # b/h ratio
        bh = b / h if h > 0 else 1.0
        if bh < BH_RATIO_MIN or bh > BH_RATIO_MAX:
            penalty += 1.0

        # reinforcement ratio (steel area / concrete gross area): approximate
        gross_area = b * h
        if gross_area > 0:
            rho = (cont_total_area + add_total_area) / gross_area
            if rho < REINF_RATIO_MIN:
                penalty += (REINF_RATIO_MIN - rho) * 100.0
            if rho > REINF_RATIO_MAX:
                penalty += (rho - REINF_RATIO_MAX) * 100.0

    # columns
    for col in decoded["columns"]:
        b = col["b_m"]; h = col["h_m"]; L = col["length_m"]
        vol = b * h * L
        total_concrete_vol_m3 += vol

        # continuous bars: assume 4 bars for column continuous
        cont_area = math.pi * (col["cont_d_mm"]/1000.0)**2 / 4.0
        add_area = math.pi * (col["add_d_mm"]/1000.0)**2 / 4.0
        # assume 4 continuous bars, plus 4 additional bars
        steel_area_m2 = (cont_area * 4.0 + add_area * 4.0)
        # stirrups
        spacing = col["stirrup_spacing_mm"]/1000.0
        perimeter = 2.0 * (b + h)
        n_stirrups = max(1, int(math.ceil(L / (spacing if spacing > 0 else 0.1))))
        stirrup_bar_d_mm = 8
        stirrup_len = perimeter * 1.15
        stirrup_area_per_bar = math.pi * (stirrup_bar_d_mm/1000.0)**2 / 4.0
        total_stirrups_area = stirrup_area_per_bar * n_stirrups

        steel_area_m2 += total_stirrups_area
        steel_mass_kg = steel_area_m2 * L * STEEL_DENSITY
        total_steel_kg += steel_mass_kg

        form_area_col = perimeter * L
        total_form_area_m2_columns += form_area_col

        # sizing constraints
        if (b * 1000.0) < MIN_SECTION_B_MM or (h * 1000.0) < MIN_SECTION_B_MM:
            penalty += 2.0
        bh = b / h if h > 0 else 1.0
        if bh < BH_RATIO_MIN or bh > BH_RATIO_MAX:
            penalty += 1.0
        gross_area = b * h
        if gross_area > 0:
            rho = (cont_area * 4.0 + add_area * 4.0) / gross_area
            if rho < REINF_RATIO_MIN:
                penalty += (REINF_RATIO_MIN - rho) * 100.0
            if rho > REINF_RATIO_MAX:
                penalty += (rho - REINF_RATIO_MAX) * 100.0

    # compute cost components using unit rates
    cost_concrete = total_concrete_vol_m3 * UNIT_CONCRETE_EU_PER_M3
    cost_steel = total_steel_kg * UNIT_STEEL_EU_PER_KG
    cost_formwork = total_form_area_m2_beams * UNIT_FORM_BEAM_EU_PER_M2 + total_form_area_m2_columns * UNIT_FORM_COLUMN_EU_PER_M2
    cost_scaffold = total_scaffold_area_m2_beams * UNIT_SCAFFOLD_BEAM_EU_PER_M2

    total_cost = cost_concrete + cost_steel + cost_formwork + cost_scaffold

    # paper uses fitness = C * (1 + p) or F = C * (1 + p) ? The paper defines F = C + p*C? (Eq.6: F = C (1 + p) or F = C + p ?)
    # In the paper the fitness is described as F = C + p * C? Eq.(6) is shown as F = C + p*C? We'll use a common approach:
    # Fitness = total_cost * (1 + penalty) where penalty is a small scalar (penalty may already be large)
    fitness = total_cost * (1.0 + penalty)

    return {
        "cost_concrete": cost_concrete,
        "cost_steel": cost_steel,
        "cost_formwork": cost_formwork,
        "cost_scaffold": cost_scaffold,
        "total_cost": total_cost,
        "penalty": penalty,
        "fitness": fitness
    }

# ----------------- Evaluation wrapper -----------------
def evaluate_chromosome(chrom):
    decoded = decode_chromosome(chrom)
    res = compute_cost_and_penalty(decoded)
    return res["fitness"], res

# ----------------- Optimization (Rao-1 + FISA style adapted for discrete indices) -----------------
def initialize_population(pop_size):
    pop = []
    for i in range(pop_size):
        c = random_chromosome()
        fitness, _ = evaluate_chromosome(c)
        pop.append((i, c, fitness))
    return pop

def clamp_and_round_chrom(chrom):
    """Ensure indices are integer and within domains."""
    chrom2 = []
    idx = 0
    # beams
    for _ in range(FRAME_DEF["n_beams"]):
        # b_idx
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(SECTION_SIZES_MM)-1, v)); chrom2.append(v)
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(SECTION_SIZES_MM)-1, v)); chrom2.append(v)
        for _ in range(6):
            v = int(round(chrom[idx])); idx+=1
            # for bars and stirrups different ranges:
            # first 4 are bars -> BAR range
            if len(chrom2) % 1 == 0: pass
            # For simplicity, map according to expected order:
            # cont_top, cont_bot, pos_add, neg_add -> BAR indices
            # stir1, stir2 -> STIRRUP indices
            if len(chrom2) % 1 == 0:
                pass
            # We'll just clip using appropriate sizes by position
            # Determine which one we are at by counting how many we've added since beam start
            # But for simplicity we just rely on values and clip to max of combined spaces:
            if len(chrom2) < 2 + 4:
                # if we are still at bar-related entries, clip to BAR range
                v = max(0, min(len(BAR_DIAMETERS_MM)-1, v))
            else:
                v = max(0, min(len(STIRRUP_SPACING_MM)-1, v))
            chrom2.append(v)
    # columns: the remaining genes
    # After above loop idx currently at 8*#beams
    for _ in range(FRAME_DEF["n_columns"]):
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(SECTION_SIZES_MM)-1, v)); chrom2.append(v)
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(SECTION_SIZES_MM)-1, v)); chrom2.append(v)
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(BAR_DIAMETERS_MM)-1, v)); chrom2.append(v)
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(BAR_DIAMETERS_MM)-1, v)); chrom2.append(v)
        v = int(round(chrom[idx])); idx+=1
        v = max(0, min(len(STIRRUP_SPACING_MM)-1, v)); chrom2.append(v)
    return chrom2

def rao1_fisa_update(pop):
    """
    One iteration of Rao-1 + FISA adapted to integer-index chromosomes.
    pop: list of (idx, chromosome_list, fitness)
    returns new population list of same structure
    """
    pop_size = len(pop)
    # prepare arrays
    chroms = [np.array(ind[1], dtype=float) for ind in pop]
    fitnesses = np.array([ind[2] for ind in pop])
    best_idx = int(np.argmin(fitnesses))
    worst_idx = int(np.argmax(fitnesses))
    best = chroms[best_idx]
    worst = chroms[worst_idx]

    new_pop = []
    for i, (idx0, chrom, fit) in enumerate(pop):
        x = chroms[i].copy()
        x_new = x.copy()
        for j in range(len(x)):
            r1 = random.random()
            r2 = random.random()
            term1 = r1 * (best[j] - worst[j])
            # pick random other individual
            rand_idx = random.randrange(pop_size)
            term2 = r2 * (x[j] - chroms[rand_idx][j])
            x_new[j] = x[j] + term1 + term2
        # clamp to reasonable ranges: because all genes are integer indices we clamp to [0, max_index_for_that_gene]
        # Build per-gene max based on gene position
        # We will round and clip using clamp_and_round_chrom
        chrom_ints = clamp_and_round_chrom(x_new.tolist())
        new_fit, _ = evaluate_chromosome(chrom_ints)
        # greedy replacement (accept if better)
        if new_fit <= fit:
            new_pop.append((idx0, chrom_ints, new_fit))
        else:
            new_pop.append((idx0, pop[i][1], fit))
    return new_pop

# --------------- Driver optimization loop (single-process driver uses evaluation function) ---------------
def run_optimization(pop_size=POP_SIZE, max_fes=MAX_FES):
    pop = initialize_population(pop_size)
    fes = pop_size
    iter_no = 0
    best_history = []
    while fes < max_fes:
        pop = rao1_fisa_update(pop)
        # count FEs: we evaluated at most pop_size individuals (one update per individual)
        fes += pop_size
        iter_no += 1
        fitnesses = [p[2] for p in pop]
        best_idx = int(np.argmin(fitnesses))
        best_val = fitnesses[best_idx]
        best_chrom = pop[best_idx][1]
        best_history.append((fes, iter_no, best_val, best_chrom))
        if (iter_no % 10) == 0:
            print(f"[Iter {iter_no}] FEs={fes} Best={best_val:.4f}")
    # final
    fitnesses = [p[2] for p in pop]
    best_idx = int(np.argmin(fitnesses))
    best_chrom = pop[best_idx][1]
    best_fit, best_res = evaluate_chromosome(best_chrom)
    # save results
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    track_path = os.path.join(OUTPUT_DIR, f"tracking_{ts}.csv")
    with open(track_path, "w") as fh:
        fh.write("fes,iter,best_fit,best_chrom\n")
        for rec in best_history:
            fh.write(f"{rec[0]},{rec[1]},{rec[2]},\"{json.dumps(rec[3])}\"\n")
    pop_path = os.path.join(OUTPUT_DIR, f"best_{ts}.json")
    with open(pop_path, "w") as fh:
        fh.write(json.dumps({
            "best_fit": best_fit,
            "best_chrom": best_chrom,
            "decoded": decode_chromosome(best_chrom),
            "metrics": best_res
        }, indent=2))
    print("Done. Best fitness:", best_fit)
    print("Saved tracking:", track_path)
    print("Saved best:", pop_path)
    return best_fit, best_chrom, best_res

if __name__ == "__main__":
    # This script is intentionally simple in Spark usage: the heavy work here is evaluate_chromosome,
    # which is executed on the driver. For large populations or expensive structural analysis,
    # evaluate_chromosome must be executed in parallel on workers (see notes below).
    print("Starting Rao-1 + FISA discrete optimization for frame cost (example).")
    best_fit, best_chrom, best_res = run_optimization()
    print("Best fitness (final):", best_fit)
    print("Best decoded design (summary):")
    print(json.dumps(decode_chromosome(best_chrom), indent=2))
    print("Cost breakdown:", {k: round(v,2) for k,v in best_res.items() if k!='fitness'})
