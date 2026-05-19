import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
model = AutoModel.from_pretrained('facebook/esm2_t12_35M_UR50D')

print(f'Parameter count: {model.num_parameters():,}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = model.to(device)

peptide = 'SIINFEKL'
inputs = tokenizer(peptide, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print(f'Embedding shape: {outputs.last_hidden_state.shape}')
print('ESM-2 working on GPU')