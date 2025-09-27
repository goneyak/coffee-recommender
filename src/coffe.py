import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 0) Path setup
# -------------------------------------------------------
os.makedirs("../outputs", exist_ok=True)

# -------------------------------------------------------
# 1) Search / Neighbor extraction
# -------------------------------------------------------
def find_coffee_indices(df, name_keyword, roaster_keyword=None, top=10):
    """Case-insensitive partial match search by coffee name (required) and roaster (optional)."""
    name_kw = str(name_keyword).lower()
    name_mask = df["Coffee Name"].astype(str).str.lower().str.contains(name_kw, na=False)
    if roaster_keyword:
        roast_kw = str(roaster_keyword).lower()
        roaster_mask = df["Roaster"].astype(str).str.lower().str.contains(roast_kw, na=False)
        mask = name_mask & roaster_mask
    else:
        mask = name_mask
    idxs = df.index[mask].tolist()
    return idxs[:top]

def topk_neighbors(sim_matrix, ref_idx, k=5, candidate_mask=None):
    """Return top-k indices based on similarity matrix (optional: restrict with candidate mask)."""
    s = sim_matrix[ref_idx].copy()
    if candidate_mask is not None:
        if len(candidate_mask) != len(s):
            raise ValueError("candidate_mask length mismatch.")
        s[~candidate_mask] = -np.inf
    order = np.argsort(s)[::-1]
    order = order[order != ref_idx]
    return order[:k], s[order[:k]]

# -------------------------------------------------------
# 2) Comparison tables / summary statistics
# -------------------------------------------------------
def rec_table(df, idxs, sims=None, extra_cols=None):
    """Build a recommendation table with metadata and optional similarity scores."""
    cols = ["Roaster","Coffee Name","coffee_origin_adj","continent","Rating","roast_lv_adj"]
    if extra_cols:
        cols = cols + [c for c in extra_cols if c not in cols]
    out = df.loc[idxs, cols].copy()
    if sims is not None:
        out = out.assign(similarity=np.round(sims, 6))
    return out.reset_index(drop=True)

def compare_models(df, sim_flavor, sim_enriched, ref_idx, k=5,
                   same_origin=False, same_continent=False, roast_tolerance=None):
    """
    Compare Flavor-only vs Enriched recommendations.
    Optional filters:
      - same_origin: restrict candidates to same origin
      - same_continent: restrict candidates to same continent
      - roast_tolerance: restrict candidates to coffees within +/- tolerance of roast_lv_adj
    """
    N = len(df)
    mask = np.ones(N, dtype=bool)
    if same_origin:
        ref_origin = str(df.loc[ref_idx, "coffee_origin_adj"]).lower()
        mask &= df["coffee_origin_adj"].astype(str).str.lower().eq(ref_origin).values
    if same_continent and "continent" in df.columns:
        ref_cont = str(df.loc[ref_idx, "continent"]).lower()
        mask &= df["continent"].astype(str).str.lower().eq(ref_cont).values
    if roast_tolerance is not None:
        ref_roast = df.loc[ref_idx, "roast_lv_adj"]
        mask &= (df["roast_lv_adj"] - ref_roast).abs().le(roast_tolerance).values

    # Flavor-only
    idx_f, sims_f = topk_neighbors(sim_flavor, ref_idx, k=k)
    tbl_f = rec_table(df, idx_f, sims_f)

    # Enriched (apply mask)
    idx_e, sims_e = topk_neighbors(sim_enriched, ref_idx, k=k, candidate_mask=mask)
    tbl_e = rec_table(df, idx_e, sims_e)

    # Overlap / Jaccard index
    set_f, set_e = set(idx_f), set(idx_e)
    overlap = sorted(list(set_f & set_e))
    jaccard = len(set_f & set_e) / len(set_f | set_e) if (set_f | set_e) else 0.0

    summary = {
        "ref": df.loc[ref_idx, ["Roaster","Coffee Name","coffee_origin_adj","continent","Rating","roast_lv_adj"]].to_dict(),
        "overlap_indices": overlap,
        "jaccard": jaccard
    }
    return tbl_f, tbl_e, summary

def explain_neighbors(df, ref_idx, nbr_idxs):
    """Show flavor differences (absolute deltas) between reference and neighbors."""
    feats = ["Aroma","Acidity_final","Body","Flavor","Aftertaste"]
    ref = df.loc[ref_idx, feats].astype(float).values
    rows = []
    for j in nbr_idxs:
        nb = df.loc[j, feats].astype(float).values
        diff = np.abs(ref - nb)
        rows.append({
            "Roaster": df.loc[j, "Roaster"],
            "Coffee Name": df.loc[j, "Coffee Name"],
            "Origin": df.loc[j, "coffee_origin_adj"],
            "Continent": df.loc[j, "continent"] if "continent" in df.columns else None,
            "Roast": df.loc[j, "roast_lv_adj"],
            "Rating": df.loc[j, "Rating"],
            "Flavor L1 Δ": float(diff.sum()),
            "Acidity Δ": float(abs(ref[1] - nb[1])),
            "Body Δ": float(abs(ref[2] - nb[2])),
        })
    return pd.DataFrame(rows).sort_values("Flavor L1 Δ").reset_index(drop=True)

# -------------------------------------------------------
# 3) Radar chart (5D flavor comparison)
# -------------------------------------------------------
FLAVOR_COLS = ["Aroma","Acidity_final","Body","Flavor","Aftertaste"]

def plot_radar(df, coffee_indices, labels=None, title="Flavor DNA"):
    """Plot a radar chart comparing flavor profiles across coffees."""
    feats = FLAVOR_COLS
    n = len(feats)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    styles  = ["solid","dashed","dashdot","dotted"]

    for idx, ci in enumerate(coffee_indices):
        vals = df.loc[ci, feats].astype(float).values
        vals = np.nan_to_num(vals).tolist()
        vals.append(vals[0])
        label = labels[idx] if labels and idx < len(labels) else f"#{ci}"
        ax.plot(angles, vals, linewidth=2, color=colors[idx%4], linestyle=styles[idx%4], label=label)
        ax.fill(angles, vals, alpha=0.08, color=colors[idx%4])

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(feats)
    ax.set_rlim(0, 10)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    plt.tight_layout()
    return fig

def compare_radar_flavor_vs_enriched(df, sim_flavor, sim_enriched, ref_idx, k=3,
                                     same_origin=False, same_continent=False, roast_tolerance=None,
                                     save_prefix="../outputs/radar"):
    """Draw and save radar charts for Flavor-only vs Enriched top-k neighbors."""
    # Candidate mask
    N = len(df)
    mask = np.ones(N, dtype=bool)
    if same_origin:
        ref_origin = str(df.loc[ref_idx, "coffee_origin_adj"]).lower()
        mask &= df["coffee_origin_adj"].astype(str).str.lower().eq(ref_origin).values
    if same_continent and "continent" in df.columns:
        ref_cont = str(df.loc[ref_idx, "continent"]).lower()
        mask &= df["continent"].astype(str).str.lower().eq(ref_cont).values
    if roast_tolerance is not None:
        ref_roast = df.loc[ref_idx, "roast_lv_adj"]
        mask &= (df["roast_lv_adj"] - ref_roast).abs().le(roast_tolerance).values

    # Flavor-only
    idx_f, _ = topk_neighbors(sim_flavor, ref_idx, k=k)
    fig1 = plot_radar(df, [ref_idx] + idx_f.tolist(),
                      labels=["Reference"] + [f"Rec{i+1}" for i in range(k)],
                      title="Flavor-only — Flavor DNA")
    fig1.savefig(f"{save_prefix}_flavor.png", dpi=160)

    # Enriched (with mask)
    idx_e, _ = topk_neighbors(sim_enriched, ref_idx, k=k, candidate_mask=mask)
    fig2 = plot_radar(df, [ref_idx] + idx_e.tolist(),
                      labels=["Reference"] + [f"Rec{i+1}" for i in range(k)],
                      title="Enriched — Flavor DNA")
    fig2.savefig(f"{save_prefix}_enriched.png", dpi=160)
    plt.show()

# -------------------------------------------------------
# 4) One-shot demo (keyword → reference → comparison)
# -------------------------------------------------------
def run_demo_compare(df, sim_flavor, sim_enriched, name_keyword,
                     roaster_keyword=None, k=5,
                     same_origin=False, same_continent=False, roast_tolerance=None):
    """
    Example:
        run_demo_compare(df, sim_flavor, sim_enriched, "santos", k=5, same_origin=True)
    """
    cand = find_coffee_indices(df, name_keyword, roaster_keyword=roaster_keyword, top=1)
    if not cand:
        raise ValueError(f"No coffee matched keyword '{name_keyword}'")
    ref_idx = cand[0]

    print("Reference:")
    display(df.loc[[ref_idx], ["Roaster","Coffee Name","coffee_origin_adj","continent","Rating","roast_lv_adj"]])

    tbl_f, tbl_e, summary = compare_models(
        df, sim_flavor, sim_enriched, ref_idx, k=k,
        same_origin=same_origin, same_continent=same_continent, roast_tolerance=roast_tolerance
    )
    print("\n=== Flavor-only Top-k ===")
    display(tbl_f)
    print("\n=== Enriched Top-k ===")
    display(tbl_e)
    print("\n=== Overlap / Jaccard ===")
    print(summary)

    # Explanation table (flavor deltas)
    print("\n--- Explain (Flavor-only) ---")
    display(explain_neighbors(df, ref_idx, tbl_f.index.tolist()))
    print("\n--- Explain (Enriched) ---")
    display(explain_neighbors(df, ref_idx, tbl_e.index.tolist()))

    # Radar charts (save top-3)
    compare_radar_flavor_vs_enriched(
        df, sim_flavor, sim_enriched, ref_idx, k=min(3, k),
        same_origin=same_origin, same_continent=same_continent, roast_tolerance=roast_tolerance
    )
