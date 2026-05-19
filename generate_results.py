import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)
os.makedirs('tables',  exist_ok=True)

# ── Load all results ───────────────────────────────────────────────────────
baseline_df  = pd.read_csv('results/baseline_results.csv', index_col=0)
fed_rounds   = pd.read_csv('results/federated_rounds.csv')
iid_noniid   = pd.read_csv('results/ablation_iid_vs_noniid.csv')
lora_rank    = pd.read_csv('results/ablation_lora_rank.csv')

print("All result files loaded successfully.")
print(f"Baseline models: {list(baseline_df.index)}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Main results
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Table 1 — Main results...")
table1 = baseline_df[['auc_roc','auc_pr','f1','mcc']].copy()
table1.columns = ['AUC-ROC','AUC-PR','F1','MCC']
table1 = table1.round(4)
table1.to_csv('tables/table1_main_results.csv')
print(table1.to_string())

# ══════════════════════════════════════════════════════════════════════════
# TABLE 2 — IID vs Non-IID
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Table 2 — IID vs Non-IID...")
table2 = pd.DataFrame({
    'Round':          iid_noniid['round'],
    'IID AUC-ROC':    iid_noniid['iid_auc'].round(4),
    'NonIID AUC-ROC': iid_noniid['noniid_auc'].round(4),
    'Gap':            (iid_noniid['iid_auc'] - iid_noniid['noniid_auc']).round(4)
})
table2.to_csv('tables/table2_iid_vs_noniid.csv', index=False)
print(f"IID best:    {iid_noniid['iid_auc'].max():.4f} "
      f"at round {iid_noniid['iid_auc'].idxmax()+1}")
print(f"NonIID best: {iid_noniid['noniid_auc'].max():.4f} "
      f"at round {iid_noniid['noniid_auc'].idxmax()+1}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 3 — Per-client AUC: local only vs federated global
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Table 3 — Per-client comparison...")
client_names = ['Coronavirus','Parasite','Human','Flavivirus','Bacteria']
local_aucs   = [
    baseline_df.loc['Single_Client1_Coronavirus', 'auc_roc'],
    baseline_df.loc['Single_Client2_Parasite',    'auc_roc'],
    baseline_df.loc['Single_Client3_Human',        'auc_roc'],
    baseline_df.loc['Single_Client4_Flavivirus',   'auc_roc'],
    baseline_df.loc['Single_Client5_Bacteria',     'auc_roc'],
]
fed_auc = baseline_df.loc['Federated_LoRA', 'auc_roc']
table3  = pd.DataFrame({
    'Client':         client_names,
    'Local Only AUC': [round(a, 4) for a in local_aucs],
    'Federated AUC':  [round(fed_auc, 4)] * 5,
    'Improvement':    [round(fed_auc - a, 4) for a in local_aucs]
})
table3.to_csv('tables/table3_per_client.csv', index=False)
print(table3.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# TABLE 4 — LoRA rank ablation
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Table 4 — LoRA rank ablation...")
lora_rank['comm_mb'] = (lora_rank['lora_params'] * 4 / 1e6).round(3)
lora_rank.to_csv('tables/table4_lora_rank.csv', index=False)
print(lora_rank.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# TABLE 5 — Communication cost
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Table 5 — Communication cost...")
full_params = 33_917_525
table5 = pd.DataFrame({
    'Method':            ['Full fine-tuning', 'LoRA r=2', 'LoRA r=4',
                          'LoRA r=8', 'LoRA r=16'],
    'Params_per_round':  [full_params, 277922, 324002, 416162, 600482],
    'Size_MB':           [round(full_params*4/1e6, 1),
                          round(277922*4/1e6, 2),
                          round(324002*4/1e6, 2),
                          round(416162*4/1e6, 2),
                          round(600482*4/1e6, 2)],
    'Reduction':         ['1x (baseline)',
                          f"{full_params/277922:.0f}x smaller",
                          f"{full_params/324002:.0f}x smaller",
                          f"{full_params/416162:.0f}x smaller",
                          f"{full_params/600482:.0f}x smaller"]
})
table5.to_csv('tables/table5_communication_cost.csv', index=False)
print(table5.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Convergence plot
# FIX: use set_xticks(range(1,21)) so rounds show as integers 1–20
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 2 — Convergence plot...")
fig, ax = plt.subplots(figsize=(10, 6))
rounds  = iid_noniid['round']

ax.plot(rounds, iid_noniid['iid_auc'],    'b-o', markersize=4,
        linewidth=2, label='IID Federated')
ax.plot(rounds, iid_noniid['noniid_auc'], 'r-s', markersize=4,
        linewidth=2, label='Non-IID Federated')
ax.axhline(y=baseline_df.loc['Centralised_LoRA','auc_roc'],
           color='green', linestyle='--', linewidth=2,
           label=f"Centralised Upper Bound "
                 f"({baseline_df.loc['Centralised_LoRA','auc_roc']:.4f})")
ax.axhline(y=baseline_df.loc['Frozen_ESM2','auc_roc'],
           color='orange', linestyle=':', linewidth=2,
           label=f"Frozen ESM-2 "
                 f"({baseline_df.loc['Frozen_ESM2','auc_roc']:.4f})")

ax.set_xlabel('Federated Round', fontsize=13)
ax.set_ylabel('AUC-ROC (Central Test Set)', fontsize=13)
ax.set_title('Federated Training Convergence: IID vs Non-IID', fontsize=14)
ax.legend(fontsize=11)
ax.set_xticks(range(1, 21))          # FIX: integer ticks 1 to 20
ax.set_xlim(0.5, 20.5)               # small padding so markers aren't clipped
ax.set_ylim(0.60, 0.85)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure2_convergence.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figures/figure2_convergence.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — LoRA rank ablation plot
# FIX: use equally spaced positions [0,1,2,3] so bars and line are symmetric
#      label them as actual rank values [2,4,8,16]
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 — LoRA rank ablation...")
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Use fixed equal positions instead of actual rank values
positions   = [0, 1, 2, 3]
rank_labels = ['2', '4', '8', '16']
aucs        = lora_rank['best_auc'].tolist()
comm        = (lora_rank['lora_params'] * 4 / 1e6).tolist()

ax2.bar(positions, comm, alpha=0.3, color='red', width=0.6,
        label='Communication (MB)')
ax1.plot(positions, aucs, 'b-o', markersize=8, linewidth=2,
         label='Best AUC-ROC')

ax1.set_xlabel('LoRA Rank (r)', fontsize=13)
ax1.set_ylabel('Best AUC-ROC', fontsize=13, color='blue')
ax2.set_ylabel('LoRA Params Communicated (MB per round)', fontsize=11,
               color='red')
ax1.set_title('LoRA Rank: Performance vs Communication Cost', fontsize=14)
ax1.set_xticks(positions)
ax1.set_xticklabels(rank_labels, fontsize=12)
ax1.set_xlim(-0.5, 3.5)             # equal padding on both sides

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure3_lora_rank.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figures/figure3_lora_rank.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Per-client bar chart (unchanged)
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4 — Per-client bar chart...")
fig, ax = plt.subplots(figsize=(10, 6))
x       = np.arange(len(client_names))
width   = 0.35
bars1   = ax.bar(x - width/2, local_aucs,    width, label='Local Only',
                 color='steelblue', alpha=0.8)
bars2   = ax.bar(x + width/2, [fed_auc]*5,   width, label='Federated Global',
                 color='coral',     alpha=0.8)

ax.set_xlabel('Client', fontsize=13)
ax.set_ylabel('AUC-ROC (Central Test Set)', fontsize=13)
ax.set_title('Per-Client: Local Training vs Federated Global Model', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(client_names, fontsize=11)
ax.set_ylim(0.45, 0.90)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('figures/figure4_per_client.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figures/figure4_per_client.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Main results bar chart (unchanged)
# ══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5 — Main results comparison...")
model_order = [
    'Random',
    'Single_Client1_Coronavirus',
    'Single_Client2_Parasite',
    'Single_Client3_Human',
    'Single_Client4_Flavivirus',
    'Single_Client5_Bacteria',
    'Frozen_ESM2',
    'Federated_LoRA',
    'Centralised_LoRA'
]
labels_short = [
    'Random',
    'Single\nCoronavirus',
    'Single\nParasite',
    'Single\nHuman',
    'Single\nFlavivirus',
    'Single\nBacteria',
    'Frozen\nESM-2',
    'Federated\nLoRA',
    'Centralised\nLoRA'
]
colors    = ['grey'] + ['steelblue']*5 + ['orange', 'coral', 'green']
aucs_main = [baseline_df.loc[m, 'auc_roc'] for m in model_order]

fig, ax = plt.subplots(figsize=(13, 6))
bars    = ax.bar(range(len(model_order)), aucs_main, color=colors, alpha=0.85)
ax.set_xticks(range(len(model_order)))
ax.set_xticklabels(labels_short, fontsize=10)
ax.set_ylabel('AUC-ROC (Central Test Set)', fontsize=13)
ax.set_title('Model Comparison: All Baselines vs Federated System', fontsize=14)
ax.set_ylim(0.40, 0.88)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.74, color='purple', linestyle='--', linewidth=1.5,
           label='BepiPred-3.0 benchmark (0.74)')
ax.legend(fontsize=10)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('figures/figure5_main_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figures/figure5_main_comparison.png")

print("\n✅ All tables and figures generated.")
print("   Tables: tables/")
print("   Figures: figures/")