"""
Multi-Structure Comparison: PrxteinMPNN vs ColabDesign

This script tests both implementations across 5+ structures of different lengths
to get robust statistical comparison.
"""

# Structure selection (diverse lengths)
TEST_STRUCTURES = [
    {"pdb": "1L2Y", "name": "Trp-cage", "length": 20, "description": "Very small miniprotein"},
    {"pdb": "5TRV", "name": "CLN025", "length": 10, "description": "β-hairpin"},
    {"pdb": "1CRN", "name": "Crambin", "length": 46, "description": "Small α-helical"},
    {"pdb": "1UBQ", "name": "Ubiquitin", "length": 76, "description": "β-grasp fold"},
    {"pdb": "2CI2", "name": "Chymotrypsin Inhibitor", "length": 64, "description": "α/β protein"},
    {"pdb": "1ROP", "name": "ROP protein", "length": 63, "description": "α-helical dimer"},
]

# Add this cell to the notebook after model loading

cell_multi_structure_download = """
# Download Multiple Test Structures
print("="*80)
print("DOWNLOADING MULTIPLE TEST STRUCTURES")
print("="*80)

test_structures = [
    {"pdb": "1L2Y", "name": "Trp-cage", "expected_length": 20},
    {"pdb": "5TRV", "name": "CLN025", "expected_length": 10},
    {"pdb": "1CRN", "name": "Crambin", "expected_length": 46},
    {"pdb": "1UBQ", "name": "Ubiquitin", "expected_length": 76},
    {"pdb": "2CI2", "name": "Chymotrypsin Inhibitor", "expected_length": 64},
    {"pdb": "1ROP", "name": "ROP protein", "expected_length": 63},
]

import os

for struct in test_structures:
    pdb_file = f"{struct['pdb'].lower()}.pdb"
    if not os.path.exists(pdb_file):
        !wget -q https://files.rcsb.org/download/{struct['pdb']}.pdb -O {pdb_file}
        print(f"✅ Downloaded {struct['name']} ({struct['pdb']}) - {struct['expected_length']} residues")
    else:
        print(f"✓  {struct['name']} ({struct['pdb']}) already exists")
"""

cell_multi_structure_test = """
# Test All Structures
print("\\n" + "="*80)
print("RUNNING COMPARISON ACROSS ALL STRUCTURES")
print("="*80)

import pandas as pd

all_results = []

for struct in test_structures:
    pdb_file = f"{struct['pdb'].lower()}.pdb"
    print(f"\\n{'='*80}")
    print(f"Testing: {struct['name']} ({struct['pdb']}) - ~{struct['expected_length']} residues")
    print(f"{'='*80}")

    try:
        # Load structure for PrxteinMPNN
        protein_tuple = next(parse_input(pdb_file))
        protein = Protein.from_tuple(protein_tuple)
        actual_length = int(protein.mask.sum())

        # Load for ColabDesign
        colab_model.prep_inputs(pdb_filename=pdb_file)

        print(f"Actual sequence length: {actual_length}")

        # Test 1: Unconditional scoring
        key = jax.random.PRNGKey(42)
        _, prx_uncond = prxtein_model(
            protein.coordinates, protein.mask, protein.residue_index,
            protein.chain_index, "unconditional", prng_key=key
        )
        col_uncond = colab_model.get_unconditional_logits()

        prx_uncond_rec = float((prx_uncond.argmax(-1) == protein.aatype).sum() / protein.mask.sum())
        col_uncond_rec = float((col_uncond.argmax(-1) == protein.aatype).sum() / protein.mask.sum())
        uncond_agree = float((prx_uncond.argmax(-1) == col_uncond.argmax(-1)).sum() / protein.mask.sum())

        # Test 2: Conditional scoring
        one_hot = jax.nn.one_hot(protein.aatype, 21)
        _, prx_cond = prxtein_model(
            protein.coordinates, protein.mask, protein.residue_index,
            protein.chain_index, "conditional", prng_key=key, one_hot_sequence=one_hot
        )
        col_output = colab_model.score(seq=None)
        col_cond = col_output["logits"]

        prx_cond_rec = float((prx_cond.argmax(-1) == protein.aatype).sum() / protein.mask.sum())
        col_cond_rec = float((col_cond.argmax(-1) == protein.aatype).sum() / protein.mask.sum())
        cond_agree = float((prx_cond.argmax(-1) == col_cond.argmax(-1)).sum() / protein.mask.sum())

        # Test 3: Sampling (5 samples at T=0.1)
        prxtein_sampler = make_sample_sequences(prxtein_model)
        prx_sample_recs = []
        key = jax.random.PRNGKey(42)
        for i in range(5):
            key, subkey = jax.random.split(key)
            sampled, _, _ = prxtein_sampler(
                subkey, protein.coordinates, protein.mask,
                protein.residue_index, protein.chain_index, temperature=jnp.array(0.1)
            )
            rec = float((sampled == protein.aatype).sum() / protein.mask.sum())
            prx_sample_recs.append(rec)

        col_sample_output = colab_model.sample(batch=5, temperature=0.1)
        col_sample_recs = col_sample_output["seqid"]

        prx_sample_mean = np.mean(prx_sample_recs)
        col_sample_mean = np.mean(col_sample_recs)

        # Logits correlation and cosine similarity
        prx_flat = np.array(prx_uncond.reshape(-1))
        col_flat = np.array(col_uncond.reshape(-1))
        correlation = np.corrcoef(prx_flat, col_flat)[0, 1]

        # Cosine similarity (1 - cosine distance)
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = cosine_similarity(prx_flat.reshape(1, -1), col_flat.reshape(1, -1))[0, 0]

        # Alternative: using scipy (cosine distance = 1 - cosine similarity)
        # from scipy.spatial.distance import cosine
        # cosine_dist = cosine(prx_flat, col_flat)
        # cosine_sim = 1 - cosine_dist

        result = {
            "PDB": struct['pdb'],
            "Name": struct['name'],
            "Length": actual_length,
            "Prx_Uncond": prx_uncond_rec,
            "Col_Uncond": col_uncond_rec,
            "Uncond_Agree": uncond_agree,
            "Prx_Cond": prx_cond_rec,
            "Col_Cond": col_cond_rec,
            "Cond_Agree": cond_agree,
            "Prx_Sample": prx_sample_mean,
            "Col_Sample": col_sample_mean,
            "Logits_Corr": correlation,
            "Cosine_Sim": cosine_sim,
        }
        all_results.append(result)

        print(f"\\nResults for {struct['name']}:")
        print(f"  Unconditional: Prx={prx_uncond_rec:.1%}, Col={col_uncond_rec:.1%}, Agree={uncond_agree:.1%}")
        print(f"  Conditional:   Prx={prx_cond_rec:.1%}, Col={col_cond_rec:.1%}, Agree={cond_agree:.1%}")
        print(f"  Sampling:      Prx={prx_sample_mean:.1%}, Col={col_sample_mean:.1%}")
        print(f"  Pearson Corr:  {correlation:.4f}")
        print(f"  Cosine Sim:    {cosine_sim:.4f}")

    except Exception as e:
        print(f"❌ Error processing {struct['name']}: {e}")
        import traceback
        traceback.print_exc()

# Create results DataFrame
df_results = pd.DataFrame(all_results)
print(f"\\n{'='*80}")
print("AGGREGATE RESULTS ACROSS ALL STRUCTURES")
print(f"{'='*80}")
print(df_results.to_string(index=False))
"""

cell_aggregate_analysis = """
# Aggregate Statistics
print(f"\\n{'='*80}")
print("AGGREGATE STATISTICS")
print(f"{'='*80}")

print(f"\\nNumber of structures tested: {len(df_results)}")
print(f"Length range: {df_results['Length'].min()}-{df_results['Length'].max()} residues")

print(f"\\n📊 UNCONDITIONAL SCORING (Mean ± Std):")
print(f"  PrxteinMPNN:  {df_results['Prx_Uncond'].mean():.1%} ± {df_results['Prx_Uncond'].std():.1%}")
print(f"  ColabDesign:  {df_results['Col_Uncond'].mean():.1%} ± {df_results['Col_Uncond'].std():.1%}")
print(f"  Agreement:    {df_results['Uncond_Agree'].mean():.1%} ± {df_results['Uncond_Agree'].std():.1%}")
print(f"  Expected:     35-65%")

print(f"\\n📊 CONDITIONAL SELF-SCORING (Mean ± Std):")
print(f"  PrxteinMPNN:  {df_results['Prx_Cond'].mean():.1%} ± {df_results['Prx_Cond'].std():.1%}")
print(f"  ColabDesign:  {df_results['Col_Cond'].mean():.1%} ± {df_results['Col_Cond'].std():.1%}")
print(f"  Agreement:    {df_results['Cond_Agree'].mean():.1%} ± {df_results['Cond_Agree'].std():.1%}")
print(f"  Expected:     >85%")

print(f"\\n📊 SAMPLING AT T=0.1 (Mean ± Std):")
print(f"  PrxteinMPNN:  {df_results['Prx_Sample'].mean():.1%} ± {df_results['Prx_Sample'].std():.1%}")
print(f"  ColabDesign:  {df_results['Col_Sample'].mean():.1%} ± {df_results['Col_Sample'].std():.1%}")
print(f"  Expected:     35-65%")

print(f"\\n📊 LOGITS SIMILARITY METRICS (Mean ± Std):")
print(f"  Pearson Corr: {df_results['Logits_Corr'].mean():.4f} ± {df_results['Logits_Corr'].std():.4f}")
print(f"  Cosine Sim:   {df_results['Cosine_Sim'].mean():.4f} ± {df_results['Cosine_Sim'].std():.4f}")
print(f"  Expected:     >0.90 for both")

# Length dependency analysis
print(f"\\n📊 LENGTH DEPENDENCY:")
print(f"  Correlation(Length, Prx_Uncond):   {df_results[['Length', 'Prx_Uncond']].corr().iloc[0,1]:.3f}")
print(f"  Correlation(Length, Agreement):    {df_results[['Length', 'Uncond_Agree']].corr().iloc[0,1]:.3f}")
print(f"  Correlation(Length, Pearson Corr): {df_results[['Length', 'Logits_Corr']].corr().iloc[0,1]:.3f}")
print(f"  Correlation(Length, Cosine Sim):   {df_results[['Length', 'Cosine_Sim']].corr().iloc[0,1]:.3f}")
"""

cell_visualization = """
# Visualize Results Across Structures
import seaborn as sns

# Define amino acid chemical types for coloring
AA_CHEMICAL_TYPES = {
    # Hydrophobic/Nonpolar
    'hydrophobic': [0, 6, 9, 11, 13, 19],  # A, G, I, L, M, V
    # Aromatic
    'aromatic': [7, 17, 20],  # F, W, Y
    # Polar uncharged
    'polar': [4, 14, 16, 18],  # C, N, Q, S, T
    # Positively charged
    'positive': [10, 8],  # H, K, R
    # Negatively charged
    'negative': [5, 6],  # D, E
    # Proline (special)
    'proline': [15],  # P
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Unconditional Recovery
axes[0, 0].bar(np.arange(len(df_results)), df_results['Prx_Uncond'], alpha=0.6, label='PrxteinMPNN')
axes[0, 0].bar(np.arange(len(df_results)), df_results['Col_Uncond'], alpha=0.6, label='ColabDesign')
axes[0, 0].axhline(0.35, color='g', linestyle='--', alpha=0.5)
axes[0, 0].axhline(0.65, color='g', linestyle='--', alpha=0.5, label='Expected range')
axes[0, 0].set_ylabel('Recovery')
axes[0, 0].set_title('Unconditional Recovery')
axes[0, 0].set_xticks(np.arange(len(df_results)))
axes[0, 0].set_xticklabels(df_results['PDB'], rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Conditional Recovery
axes[0, 1].bar(np.arange(len(df_results)), df_results['Prx_Cond'], alpha=0.6, label='PrxteinMPNN')
axes[0, 1].bar(np.arange(len(df_results)), df_results['Col_Cond'], alpha=0.6, label='ColabDesign')
axes[0, 1].axhline(0.85, color='r', linestyle='--', label='Expected >85%')
axes[0, 1].set_ylabel('Recovery')
axes[0, 1].set_title('Conditional Self-Scoring')
axes[0, 1].set_xticks(np.arange(len(df_results)))
axes[0, 1].set_xticklabels(df_results['PDB'], rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Agreement
axes[0, 2].bar(np.arange(len(df_results)), df_results['Uncond_Agree'], color='purple', alpha=0.6)
axes[0, 2].axhline(0.80, color='r', linestyle='--', label='Expected >80%')
axes[0, 2].set_ylabel('Agreement')
axes[0, 2].set_title('Unconditional Agreement')
axes[0, 2].set_xticks(np.arange(len(df_results)))
axes[0, 2].set_xticklabels(df_results['PDB'], rotation=45)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Sampling Recovery
axes[1, 0].bar(np.arange(len(df_results)), df_results['Prx_Sample'], alpha=0.6, label='PrxteinMPNN')
axes[1, 0].bar(np.arange(len(df_results)), df_results['Col_Sample'], alpha=0.6, label='ColabDesign')
axes[1, 0].axhline(0.35, color='g', linestyle='--', alpha=0.5)
axes[1, 0].axhline(0.65, color='g', linestyle='--', alpha=0.5, label='Expected range')
axes[1, 0].set_ylabel('Recovery')
axes[1, 0].set_title('Sampling Recovery (T=0.1)')
axes[1, 0].set_xticks(np.arange(len(df_results)))
axes[1, 0].set_xticklabels(df_results['PDB'], rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Logits Similarity Metrics (Pearson Correlation vs Cosine Similarity)
x_pos = np.arange(len(df_results))
width = 0.35
axes[1, 1].bar(x_pos - width/2, df_results['Logits_Corr'], width,
               alpha=0.6, label='Pearson Corr', color='orange')
axes[1, 1].bar(x_pos + width/2, df_results['Cosine_Sim'], width,
               alpha=0.6, label='Cosine Sim', color='teal')
axes[1, 1].axhline(0.90, color='r', linestyle='--', label='Expected >0.90')
axes[1, 1].set_ylabel('Similarity')
axes[1, 1].set_title('Logits Similarity: Pearson vs Cosine')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(df_results['PDB'], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Length vs Recovery
axes[1, 2].scatter(df_results['Length'], df_results['Prx_Uncond'], s=100, alpha=0.6, label='Prx Uncond')
axes[1, 2].scatter(df_results['Length'], df_results['Col_Uncond'], s=100, alpha=0.6, label='Col Uncond')
axes[1, 2].scatter(df_results['Length'], df_results['Uncond_Agree'], s=100, alpha=0.6, label='Agreement')
axes[1, 2].set_xlabel('Sequence Length')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Metrics vs Sequence Length')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_structure_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n✅ Plots saved to multi_structure_comparison.png")
"""

cell_violin_plots = """
# Violin Plots for Recovery Distributions
print("\\n" + "="*80)
print("SEQUENCE RECOVERY DISTRIBUTIONS")
print("="*80)

# Prepare data for violin plots
recovery_data = []
for _, row in df_results.iterrows():
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'PrxteinMPNN',
        'Mode': 'Unconditional',
        'Recovery': row['Prx_Uncond']
    })
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'ColabDesign',
        'Mode': 'Unconditional',
        'Recovery': row['Col_Uncond']
    })
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'PrxteinMPNN',
        'Mode': 'Conditional',
        'Recovery': row['Prx_Cond']
    })
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'ColabDesign',
        'Mode': 'Conditional',
        'Recovery': row['Col_Cond']
    })
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'PrxteinMPNN',
        'Mode': 'Sampling',
        'Recovery': row['Prx_Sample']
    })
    recovery_data.append({
        'Structure': row['PDB'],
        'Implementation': 'ColabDesign',
        'Mode': 'Sampling',
        'Recovery': row['Col_Sample']
    })

import pandas as pd
df_violin = pd.DataFrame(recovery_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Unconditional Recovery Distribution
sns.violinplot(data=df_violin[df_violin['Mode'] == 'Unconditional'],
               x='Implementation', y='Recovery', ax=axes[0], inner='box')
axes[0].axhline(0.35, color='g', linestyle='--', alpha=0.5, label='Expected range')
axes[0].axhline(0.65, color='g', linestyle='--', alpha=0.5)
axes[0].set_title('Unconditional Recovery Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Recovery', fontsize=12)
axes[0].set_ylim([0, 1])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Conditional Recovery Distribution
sns.violinplot(data=df_violin[df_violin['Mode'] == 'Conditional'],
               x='Implementation', y='Recovery', ax=axes[1], inner='box')
axes[1].axhline(0.85, color='r', linestyle='--', alpha=0.5, label='Expected >85%')
axes[1].set_title('Conditional Self-Scoring Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Recovery', fontsize=12)
axes[1].set_ylim([0, 1])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Sampling Recovery Distribution
sns.violinplot(data=df_violin[df_violin['Mode'] == 'Sampling'],
               x='Implementation', y='Recovery', ax=axes[2], inner='box')
axes[2].axhline(0.35, color='g', linestyle='--', alpha=0.5, label='Expected range')
axes[2].axhline(0.65, color='g', linestyle='--', alpha=0.5)
axes[2].set_title('Sampling Recovery Distribution (T=0.1)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Recovery', fontsize=12)
axes[2].set_ylim([0, 1])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('recovery_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n✅ Violin plots saved to recovery_distributions.png")
"""

cell_logits_scatter_enhanced = """
# Enhanced Logits Scatter Plot with Chemical Property Coloring
print("\\n" + "="*80)
print("LOGITS SCATTER PLOT WITH CHEMICAL PROPERTY COLORING")
print("="*80)

# Choose a representative structure for detailed analysis (e.g., 1UBQ)
struct_idx = 3  # 1UBQ
struct = test_structures[struct_idx]
pdb_file = f\"{struct['pdb'].lower()}.pdb\"

# Reload and analyze this structure
protein_tuple = next(parse_input(pdb_file))
protein = Protein.from_tuple(protein_tuple)

# Get logits
key = jax.random.PRNGKey(42)
_, prx_logits = prxtein_model(
    protein.coordinates, protein.mask, protein.residue_index,
    protein.chain_index, \"unconditional\", prng_key=key
)
colab_model.prep_inputs(pdb_filename=pdb_file)
col_logits = colab_model.get_unconditional_logits()

# Define amino acid chemical types
AA_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']

AA_CHEMICAL_MAP = {
    # Hydrophobic/Nonpolar (brown/tan)
    0: 'Hydrophobic', 6: 'Hydrophobic', 9: 'Hydrophobic',
    11: 'Hydrophobic', 13: 'Hydrophobic', 19: 'Hydrophobic',
    # Aromatic (purple)
    7: 'Aromatic', 17: 'Aromatic', 20: 'Aromatic',
    # Polar uncharged (green)
    4: 'Polar', 14: 'Polar', 16: 'Polar', 18: 'Polar',
    # Positively charged (blue)
    10: 'Positive', 8: 'Positive', 1: 'Positive',
    # Negatively charged (red)
    5: 'Negative', 6: 'Negative',
    # Proline (yellow)
    15: 'Proline',
    # Glycine (gray)
    7: 'Glycine',
}

# Color palette
color_map = {
    'Hydrophobic': '#8B4513',  # brown
    'Aromatic': '#9370DB',     # purple
    'Polar': '#2E8B57',        # green
    'Positive': '#4169E1',     # blue
    'Negative': '#DC143C',     # red
    'Proline': '#FFD700',      # gold
    'Glycine': '#808080',      # gray
    'Unknown': '#000000',      # black
}

# Create colors array for each logit value
n_positions = int(protein.mask.sum())
colors = []
for pos in range(n_positions):
    for aa in range(21):
        chem_type = AA_CHEMICAL_MAP.get(aa, 'Unknown')
        colors.append(color_map[chem_type])

# Flatten logits
prx_flat = np.array(prx_logits[:n_positions].reshape(-1))
col_flat = np.array(col_logits[:n_positions].reshape(-1))

# Create enhanced scatter plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Scatter with chemical property coloring
axes[0].scatter(prx_flat, col_flat, c=colors, alpha=0.5, s=10, edgecolors='none')
axes[0].plot([-10, 10], [-10, 10], 'k--', linewidth=2, label='Perfect agreement', alpha=0.7)
axes[0].set_xlabel('PrxteinMPNN Logits', fontsize=12)
axes[0].set_ylabel('ColabDesign Logits', fontsize=12)
axes[0].set_title(f'Logits Comparison - {struct[\"name\"]} ({struct[\"pdb\"]})', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Add correlation info
correlation = np.corrcoef(prx_flat, col_flat)[0, 1]
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(prx_flat.reshape(1, -1), col_flat.reshape(1, -1))[0, 0]
axes[0].text(0.05, 0.95, f'Pearson: {correlation:.3f}\\nCosine: {cosine_sim:.3f}',
            transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Legend showing chemical types
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map[ctype], label=ctype)
                   for ctype in ['Hydrophobic', 'Aromatic', 'Polar', 'Positive', 'Negative', 'Proline', 'Glycine']]
axes[1].legend(handles=legend_elements, loc='center', fontsize=12, title='Amino Acid Chemical Types', title_fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.savefig(f'logits_scatter_{struct[\"pdb\"]}_chemical.png', dpi=150, bbox_inches='tight')
plt.show()

print(f\"\\n✅ Enhanced scatter plot saved to logits_scatter_{struct['pdb']}_chemical.png\")
"""

# Print the cells to add
print("Add these cells to the Colab notebook after the model loading section:")
print("\n" + "="*80)
print("CELL: Download Multiple Structures")
print("="*80)
print(cell_multi_structure_download)

print("\n" + "="*80)
print("CELL: Test All Structures")
print("="*80)
print(cell_multi_structure_test)

print("\n" + "="*80)
print("CELL: Aggregate Analysis")
print("="*80)
print(cell_aggregate_analysis)

print("\n" + "="*80)
print("CELL: Visualization")
print("="*80)
print(cell_visualization)

print("\n" + "="*80)
print("CELL: Violin Plots for Recovery Distributions")
print("="*80)
print(cell_violin_plots)

print("\n" + "="*80)
print("CELL: Enhanced Logits Scatter Plot with Chemical Coloring")
print("="*80)
print(cell_logits_scatter_enhanced)
