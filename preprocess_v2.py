import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("Loading CSV...")
df = pd.read_csv('bcell_full_v3.csv', header=[0,1], low_memory=False)
df.columns = [f"{a}|{b}" for a, b in df.columns]

# ── Clean ──────────────────────────────────────────────────────────────────
df = df[df['Epitope|Object Type'] == 'Linear peptide']
df = df[df['Assay|Qualitative Measure'].isin(['Positive', 'Negative'])]
df = df.rename(columns={
    'Epitope|Name': 'sequence',
    'Assay|Qualitative Measure': 'label_str',
    'Epitope|Source Organism': 'organism'
})
df['label'] = (df['label_str'] == 'Positive').astype(int)
df = df[['sequence', 'label', 'organism']].copy()
df = df.dropna(subset=['sequence', 'organism'])

standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
def is_valid(seq):
    return isinstance(seq, str) and set(seq.upper()).issubset(standard_aa)

df = df[df['sequence'].apply(is_valid)]
df['sequence'] = df['sequence'].str.upper()
df = df[df['sequence'].str.len().between(8, 25)]
df = df.drop_duplicates(subset=['sequence'])
print(f"Clean dataset: {len(df)} sequences")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nTop 20 organisms:\n{df['organism'].value_counts().head(20)}")

# ── Global train/test split FIRST ─────────────────────────────────────────
train_val_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)
print(f"\nCentral test set: {len(test_df)}")
print(f"Training pool: {len(train_val_df)}")

os.makedirs('data', exist_ok=True)
test_df[['sequence', 'label', 'organism']].to_csv(
    'data/central_test.csv', index=False)
print(f"Saved central_test.csv with organism column for analysis")

# ── Non-IID partitioning — group organisms into 5 clients ─────────────────
# Group by biological similarity so each client has enough data
organism_counts = train_val_df['organism'].value_counts()
print(f"\nOrganisms in training pool: {len(organism_counts)}")

# Define 5 client groups by organism
client_organism_groups = {
    'client1': [  # Coronaviruses
        'Severe acute respiratory syndrome coronavirus 2',
        'Severe acute respiratory syndrome coronavirus 2 Wuhan/Hu-1/2019',
        'SARS-CoV1',
        'SARS coronavirus Tor2',
        'Human coronavirus OC43',
        'Human coronavirus NL63',
        'Human coronavirus 229E',
        'Human coronavirus HKU1',
        'Middle East respiratory syndrome-related coronavirus',
        'Pangolin coronavirus',
    ],
    'client2': [  # Parasites
        'Trypanosoma cruzi strain CL Brener',
        'Schistosoma mansoni',
        'Onchocerca volvulus',
        'Plasmodium falciparum',
        'Leishmania donovani',
    ],
    'client3': [  # Human / self antigens
        'Homo sapiens',
        'Mus musculus',
    ],
    'client4': [  # Hepatitis / flaviviruses
        'Hepacivirus hominis',
        'Dengue virus',
        'Zika virus',
        'Hepatitis B virus',
        'West Nile virus',
    ],
    'client5': [  # Bacteria and other pathogens
        'Streptococcus pyogenes',
        'Mycobacterium tuberculosis',
        'Staphylococcus aureus',
        'Bacillus anthracis',
        'Yersinia pestis',
        'Clostridium tetani',
        'Influenza A virus',
        'Human immunodeficiency virus 1',
    ]
}

# Assign organisms to clients, everything else goes to closest group
assigned = set()
client_dfs = {}

for client_id, organisms in client_organism_groups.items():
    mask = train_val_df['organism'].isin(organisms)
    client_dfs[client_id] = train_val_df[mask].copy()
    assigned.update(organisms)
    print(f"{client_id}: {len(client_dfs[client_id])} sequences "
          f"({client_dfs[client_id]['label'].mean():.1%} positive)")

# Remaining organisms — distribute evenly to balance client sizes
remaining = train_val_df[~train_val_df['organism'].isin(assigned)].copy()
print(f"\nRemaining unassigned: {len(remaining)} sequences")
print(f"Remaining organisms: {remaining['organism'].nunique()}")

# Sort remaining organisms by size and distribute round-robin
remaining_orgs = remaining['organism'].value_counts()
client_ids = list(client_organism_groups.keys())
for idx, (org, count) in enumerate(remaining_orgs.items()):
    target_client = client_ids[idx % 5]
    org_data = remaining[remaining['organism'] == org]
    client_dfs[target_client] = pd.concat(
        [client_dfs[target_client], org_data])

print(f"\nFinal client sizes after redistribution:")
for client_id, cdf in client_dfs.items():
    pos_rate = cdf['label'].mean()
    print(f"  {client_id}: {len(cdf)} sequences, "
          f"{pos_rate:.1%} positive")

# Save client files with 80/20 local split
for client_id, cdf in client_dfs.items():
    if len(cdf) < 10:
        print(f"WARNING: {client_id} too small, skipping")
        continue
    c_train, c_val = train_test_split(
        cdf, test_size=0.2, random_state=42, stratify=cdf['label']
    )
    c_train[['sequence', 'label']].to_csv(
        f'data/{client_id}_train.csv', index=False)
    c_val[['sequence', 'label']].to_csv(
        f'data/{client_id}_val.csv', index=False)
    print(f"Saved {client_id}: train={len(c_train)}, val={len(c_val)}")

# ── IID partitioning ───────────────────────────────────────────────────────
iid_splits = np.array_split(train_val_df.sample(frac=1, random_state=42), 5)
for i, split in enumerate(iid_splits):
    c_train, c_val = train_test_split(
        split, test_size=0.2, random_state=42, stratify=split['label']
    )
    c_train[['sequence', 'label']].to_csv(
        f'data/iid_client{i+1}_train.csv', index=False)
    c_val[['sequence', 'label']].to_csv(
        f'data/iid_client{i+1}_val.csv', index=False)
    print(f"Saved iid_client{i+1}: train={len(c_train)}, val={len(c_val)}")

# ── Final accounting ───────────────────────────────────────────────────────
total_client = sum([len(client_dfs[f'client{i+1}']) for i in range(5)])
print(f"\n{'='*50}")
print(f"Total in clients:   {total_client}")
print(f"Total in test:      {len(test_df)}")
print(f"Grand total:        {total_client + len(test_df)}")
print(f"Original clean:     {len(df)}")
print(f"Unaccounted:        {len(df) - total_client - len(test_df)}")
print(f"✅ Preprocessing v2 complete")
