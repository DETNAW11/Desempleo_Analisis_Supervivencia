import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

# --- Cargar datos (mismos que ya tienes) ---
train = pd.read_csv("train_desempleo.csv")
test = pd.read_csv("test_desempleo.csv")
feature_cols = [c for c in train.columns if c not in ["Tiempo_Busqueda", "Ocupados"]]

# --- Reentrenar Cox (para tener el objeto disponible aquí) ---
cph = CoxPHFitter(penalizer=0.1)
cph.fit(train, duration_col="Tiempo_Busqueda", event_col="Ocupados")

# --- Reentrenar la red neuronal (misma arquitectura y semilla que ya usamos) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols].astype(float))

class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def cox_partial_log_likelihood(risk_scores, durations, events):
    order = torch.argsort(durations, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]
    hazard_ratio = torch.exp(risk_scores)
    log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-8)
    log_lik = risk_scores - log_cum_hazard
    return -torch.sum(log_lik * events) / torch.sum(events)

torch.manual_seed(42)
model = DeepSurv(input_dim=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
T_train_t = torch.tensor(train["Tiempo_Busqueda"].values, dtype=torch.float32)
E_train_t = torch.tensor(train["Ocupados"].values, dtype=torch.float32)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    risk_scores = model(X_train_t)
    loss = cox_partial_log_likelihood(risk_scores, T_train_t, E_train_t)
    loss.backward()
    optimizer.step()

# =========================================================
# GRÁFICA 1: Curvas de supervivencia predichas (Cox vs Red)
#            para dos perfiles representativos
# =========================================================

# Perfil A: el observado con el riesgo más bajo (según Cox) en test
# Perfil B: el observado con el riesgo más alto (según Cox) en test
risk_cox_test = cph.predict_partial_hazard(test)
idx_bajo_riesgo = risk_cox_test.idxmin()
idx_alto_riesgo = risk_cox_test.idxmax()

fig, ax = plt.subplots(figsize=(8, 5))

for idx, label, color in [(idx_bajo_riesgo, "Perfil bajo riesgo", "tab:blue"),
                            (idx_alto_riesgo, "Perfil alto riesgo", "tab:red")]:
    perfil = test.loc[[idx]]
    sf = cph.predict_survival_function(perfil)
    ax.plot(sf.index, sf.values.flatten(), label=f"Cox - {label}", color=color, linestyle="-")

ax.set_xlabel("Semanas")
ax.set_ylabel("Probabilidad de seguir desempleado")
ax.set_title("Curvas de supervivencia estimadas (Modelo de Cox)\nPerfiles de bajo y alto riesgo")
ax.legend()
ax.set_xlim(0, 72)
plt.tight_layout()
plt.savefig("curvas_supervivencia_cox.png", dpi=150)
plt.show()

# =========================================================
# GRÁFICA 2: Comparación de C-index (barra simple)
# =========================================================

fig, ax = plt.subplots(figsize=(6, 5))
modelos = ["Cox (clásico)", "Red Neuronal\n(DeepSurv)"]
c_index_valores = [0.6056, 0.5970]  # ajusta si tus valores exactos difieren
colores = ["tab:blue", "tab:orange"]

bars = ax.bar(modelos, c_index_valores, color=colores, width=0.5)
ax.axhline(0.5, color="gray", linestyle="--", label="Azar (0.5)")
ax.set_ylabel("C-index (test)")
ax.set_ylim(0.45, 0.65)
ax.set_title("Comparación de desempeño: Cox vs. Red Neuronal")

for bar, valor in zip(bars, c_index_valores):
    ax.text(bar.get_x() + bar.get_width()/2, valor + 0.005, f"{valor:.3f}",
            ha="center", fontweight="bold")

ax.legend()
plt.tight_layout()
plt.savefig("comparacion_c_index.png", dpi=150)
plt.show()