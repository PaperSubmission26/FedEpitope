"""
filter_test.py
Generates a strict de-duplicated test set by removing substring-overlapping sequences.
Usage: python filter_test.py
Output: data/central_test_strict.csv
"""
import pandas as pd

flagged = pd.read_csv('leakage_report.csv')
flagged_seqs = set(flagged['test_seq'].str.upper())

test = pd.read_csv('data/central_test.csv')
original_size = len(test)

strict = test[~test['sequence'].str.upper().isin(flagged_seqs)]
strict.to_csv('data/central_test_strict.csv', index=False)

print(f'Original test set : {original_size:,}')
print(f'Removed           : {original_size - len(strict):,} ({100*(original_size-len(strict))/original_size:.1f}%)')
print(f'Strict test set   : {len(strict):,}')
print(f'Saved to          : data/central_test_strict.csv')