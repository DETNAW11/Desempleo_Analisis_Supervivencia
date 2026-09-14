import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

# --- Cargar los mismos train/test que usamos en el Cox ---
train = pd.read_csv("train_desempleo.csv")
test = pd.read_csv("test_desempleo.csv")

feature_cols = [c for c in train.columns if c not in ["Tiempo_Busqueda", "Ocupados"]]

# Estandarizar features (importante para redes neuronales, a diferencia de Cox clásico)
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols].astype(float))
X_test = scaler.transform(test[feature_cols].astype(float))

T_train = train["Tiempo_Busqueda"].values.astype(float)
E_train = train["Ocupados"].values.astype(float)
T_test = test["Tiempo_Busqueda"].values.astype(float)
E_test = test["Ocupados"].values.astype(float)

# Convertir a tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32)
T_train_t = torch.tensor(T_train, dtype=torch.float32)
E_train_t = torch.tensor(E_train, dtype=torch.float32)

# --- Arquitectura simple (dataset pequeño -> pocas capas, dropout fuerte) ---
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- Pérdida: log-verosimilitud parcial negativa de Cox (aproximación de Breslow) ---
def cox_partial_log_likelihood(risk_scores, durations, events):
    order = torch.argsort(durations, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    hazard_ratio = torch.exp(risk_scores)
    log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
    log_lik = risk_scores - log_cum_hazard
    return -torch.sum(log_lik * events) / torch.sum(events)

# --- Entrenamiento ---
torch.manual_seed(42)
model = DeepSurv(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # weight_decay = regularización L2

n_epochs = 300
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    risk_scores = model(X_train_t)
    loss = cox_partial_log_likelihood(risk_scores, T_train_t, E_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f}")

# --- Evaluación con C-index (misma métrica que usamos en Cox, para comparación justa) ---
model.eval()
with torch.no_grad():
    risk_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

c_index_nn = concordance_index(T_test, -risk_test, E_test)
print("\nC-index de la red neuronal en test:", c_index_nn)
print("C-index del Cox clásico en test (referencia): 0.6056")