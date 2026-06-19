"""
Arquitectura FraudDNN (PyTorch) — modelo del paper.

DNN tabular multi-entrada: flujo denso (numéricas + one-hot de baja cardinalidad)
+ embeddings dedicados para campos de alta cardinalidad (emisor, región, categoría),
tres bloques residuales, un bloque de atención Squeeze-and-Excitation y una salida
sigmoide con la probabilidad de fraude.

Definición idéntica a la usada en el entrenamiento (notebooks/_common_train.py),
para que los pesos `fraud_dnn_weights.pt` carguen sin discrepancias.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_d, out_d, drop):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        self.bn = nn.BatchNorm1d(out_d)
        self.drop = nn.Dropout(drop)
        self.skip = nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity()

    def forward(self, x):
        return self.drop(torch.relu(self.bn(self.linear(x)))) + self.skip(x)


class SEBlock(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.fc1 = nn.Linear(u, max(1, u // 4))
        self.fc2 = nn.Linear(max(1, u // 4), u)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class FraudDNN(nn.Module):
    def __init__(self, n_numeric, emb_spec, units, dropout, n_res_blocks):
        super().__init__()
        self.emb_keys = list(emb_spec.keys())
        self.embeddings = nn.ModuleDict({
            f: nn.Embedding(s['n_categories'] + 1, s['embedding_dim'], padding_idx=0)
            for f, s in emb_spec.items()
        })
        cur = n_numeric + sum(s['embedding_dim'] for s in emb_spec.values())
        self.res_blocks = nn.ModuleList()
        for _ in range(n_res_blocks):
            self.res_blocks.append(ResidualBlock(cur, units, dropout))
            cur = units
        self.se = SEBlock(units)
        self.fc = nn.Linear(units, 64)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x_num, x_embs):
        ep = [self.embeddings[f](x_embs[:, i]) for i, f in enumerate(self.emb_keys)]
        x = torch.cat([x_num] + ep, dim=1)
        for b in self.res_blocks:
            x = b(x)
        return torch.sigmoid(self.out(self.drop(torch.relu(self.fc(self.se(x)))))).squeeze(-1)
