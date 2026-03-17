# Taller 02 — ML Dashboard | EAFIT 2026-1

Dashboard interactivo para predicción de costos médicos (regresión) y churn de clientes (clasificación).

## Estructura del repositorio

```
/
├── app/
│   └── app.py                  ← Dashboard principal (Streamlit)
├── notebooks/
│   ├── Taller02_Regresion_MedicalCost.ipynb
│   └── Taller02_Clasificacion_TelcoChurn.ipynb
├── data/
│   ├── insurance.csv           ← Dataset Medical Cost (Kaggle)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Dataset Telco (Kaggle)
├── requirements.txt
└── README.md
```

## Instalación local

```bash
# 1. Clonar repositorio
git clone https://github.com/<tu-usuario>/taller02-ml-eafit.git
cd taller02-ml-eafit

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar dashboard
streamlit run app/app.py
```

## Despliegue en Streamlit Cloud

1. Subir el repositorio a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. Conectar con tu cuenta de GitHub
4. Seleccionar repositorio → rama `main` → archivo `app/app.py`
5. Click en **Deploy**

## Funcionalidades del Dashboard

| Módulo | Descripción |
|---|---|
| 🏥 Regresión | Predicción de costo médico (Medical Cost Dataset) |
| 📱 Clasificación | Predicción de churn (Telco Customer Churn) |
| 📊 Métricas | Comparación de 3 modelos con CV + Test Set |
| 📈 Feature Importance | Gráfica de importancia de variables (Random Forest) |
| 🔍 Predicción Individual | Formulario interactivo con análisis de riesgo |
| 📂 Predicción por Lote | Carga de CSV + descarga de resultados |

## Modelos implementados

### Regresión (Medical Cost)
| Modelo | Hiperparámetros clave |
|---|---|
| Ridge Regression | alpha=10.0 |
| Random Forest | n_estimators=200, max_depth=20 |
| Gradient Boosting | n_estimators=200, lr=0.1, max_depth=5 |

### Clasificación (Telco Churn)
| Modelo | Hiperparámetros clave |
|---|---|
| Logistic Regression | C=1.0, class_weight='balanced' |
| Random Forest | n_estimators=200, class_weight='balanced' |
| Gradient Boosting | n_estimators=200, lr=0.1, max_depth=5 |

## Métricas de evaluación

- **Regresión:** R², MAE (USD), RMSE (USD)
- **Clasificación:** AUC-ROC, F1-Score, Accuracy

## Datasets

- [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---
*Universidad EAFIT · Maestría en Ciencia de Datos · Docente: Jorge Iván Padilla-Buriticá*
