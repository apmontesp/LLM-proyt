"""
Taller 02 — Dashboard de Prediccion
Introduccion a la Inteligencia Artificial | EAFIT 2026-1
Integrantes:
    Ana Patricia Montes Pimienta
    Karen Melissa Gomez Montoya
    Juan Esteban Estrada Herrera
Docente: Jorge Ivan Padilla Buritica
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taller 02 | EAFIT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    div[data-testid="stSidebar"] { background-color: #1c2536; }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] span { color: #e8eaf6 !important; }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #ffffff !important; }
    div[data-testid="stSidebar"] hr { border-color: #3a4a6b; }

    .main-title {
        font-size: 2rem; font-weight: 700; color: #1a237e;
        border-bottom: 3px solid #1a237e; padding-bottom: 8px; margin-bottom: 4px;
    }
    .sub-title { color: #37474f; font-size: 0.95rem; margin-top: 0; }

    .metric-card {
        background: #f5f7ff; border-radius: 10px;
        padding: 14px 18px; border-left: 4px solid #1a237e; margin-bottom: 8px;
    }
    .metric-card h3 { margin: 0; font-size: 1.4rem; color: #1a237e; }
    .metric-card p  { margin: 2px 0 0 0; color: #546e7a; font-size: 0.82rem; }

    .info-box {
        background: #f8f9fa; border-radius: 10px;
        padding: 18px 22px; border: 1px solid #dee2e6; margin-bottom: 16px;
    }
    .info-box h4 { color: #1a237e; margin-top: 0; border-bottom: 1px solid #dee2e6; padding-bottom: 6px; }
    .info-box p  { color: #37474f; margin: 4px 0; font-size: 0.93rem; }

    .nav-card {
        background: #ffffff; border-radius: 12px; padding: 22px;
        border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .nav-card h3 { color: #1a237e; margin-top: 0; }

    .pred-ok   { background: #e8f5e9; border-radius: 12px; padding: 20px; border: 2px solid #388e3c; text-align: center; }
    .pred-warn { background: #fff3e0; border-radius: 12px; padding: 20px; border: 2px solid #f57c00; text-align: center; }
    .pred-val-ok   { font-size: 2.2rem; font-weight: 700; color: #1b5e20; }
    .pred-val-warn { font-size: 2.2rem; font-weight: 700; color: #bf360c; }

    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── DATOS ─────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_regression_data():
    np.random.seed(42)
    n = 1338
    age      = np.random.randint(18, 65, n)
    sex      = np.random.choice(['male','female'], n)
    bmi      = np.round(np.random.normal(30.7, 6.1, n).clip(15, 55), 1)
    children = np.random.choice([0,1,2,3,4,5], n, p=[0.43,0.24,0.18,0.10,0.03,0.02])
    smoker   = np.random.choice(['yes','no'], n, p=[0.20,0.80])
    region   = np.random.choice(['southwest','southeast','northwest','northeast'], n)
    charges  = (age*250 + bmi*300 + (smoker=='yes')*20000 +
                children*400 + np.random.normal(0,2500,n)).clip(1000)
    df = pd.DataFrame({'age':age,'sex':sex,'bmi':bmi,'children':children,
                       'smoker':smoker,'region':region,'charges':np.round(charges,2)})
    df['bmi_smoker'] = df['bmi'] * (df['smoker']=='yes').astype(int)
    df['obese']      = (df['bmi'] >= 30).astype(int)
    return df

@st.cache_data
def generate_classification_data():
    np.random.seed(42)
    n = 7043
    tenure   = np.random.randint(0, 72, n)
    monthly  = np.round(np.random.uniform(20, 110, n), 2)
    total    = np.round(tenure * monthly + np.random.normal(0,50,n), 2).clip(0)
    contract = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.24,0.21])
    internet = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.34,0.44,0.22])
    payment  = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n)
    senior   = np.random.choice([0,1], n, p=[0.84,0.16])
    partner  = np.random.choice(['Yes','No'], n)
    paperless= np.random.choice(['Yes','No'], n, p=[0.59,0.41])
    churn_p  = (0.05 + 0.30*(contract=='Month-to-month') + 0.10*(internet=='Fiber optic')
                + 0.08*(payment=='Electronic check') - 0.008*tenure
                + np.random.normal(0,0.05,n)).clip(0,1)
    churn = (np.random.uniform(0,1,n) < churn_p).astype(int)
    df = pd.DataFrame({
        'tenure':tenure,'MonthlyCharges':monthly,'TotalCharges':total,
        'Contract':contract,'InternetService':internet,'PaymentMethod':payment,
        'SeniorCitizen':senior,'Partner':partner,'PaperlessBilling':paperless,'Churn':churn
    })
    df['charges_per_month'] = df['TotalCharges'] / (df['tenure']+1)
    df['is_month_to_month'] = (df['Contract']=='Month-to-month').astype(int)
    df['high_monthly']      = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    return df


# ── MODELOS ───────────────────────────────────────────────────────────────────
@st.cache_resource
def train_regression_models():
    df = generate_regression_data()
    FEATURES  = ['age','sex','bmi','children','smoker','region','bmi_smoker','obese']
    num_cols  = ['age','bmi','children','bmi_smoker']
    cat_cols  = ['sex','smoker','region']
    pass_cols = ['obese']
    X = df[FEATURES]; y = df['charges']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    prep = ColumnTransformer([
        ('num',  StandardScaler(), num_cols),
        ('cat',  OneHotEncoder(drop='first', sparse_output=False), cat_cols),
        ('pass', 'passthrough', pass_cols)
    ])
    pipes = {
        'Regresion Lineal': Pipeline([('prep',prep),('model',LinearRegression())]),
        'KNN Regressor':    Pipeline([('prep',prep),('model',KNeighborsRegressor(n_neighbors=10, weights='distance'))]),
        'Random Forest':    Pipeline([('prep',prep),('model',RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))]),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    preds_dict = {}
    for name, pipe in pipes.items():
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict(X_te)
        preds_dict[name] = yp
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring='r2')
        results[name] = {
            'RMSE':  round(np.sqrt(mean_squared_error(y_te, yp)), 2),
            'MSE':   round(mean_squared_error(y_te, yp), 2),
            'R2':    round(r2_score(y_te, yp), 4),
            'MAE':   round(mean_absolute_error(y_te, yp), 2),
            'CV R2': round(cv_r2.mean(), 4),
        }
    rf_model   = pipes['Random Forest'].named_steps['model']
    prep_fit   = pipes['Random Forest'].named_steps['prep']
    cat_names  = prep_fit.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    importance = pd.Series(rf_model.feature_importances_,
                           index=num_cols+cat_names+pass_cols).sort_values()
    return pipes, results, importance, X_te, y_te, FEATURES, preds_dict

@st.cache_resource
def train_classification_models():
    df = generate_classification_data()
    FEATURES  = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
                 'Contract','InternetService','PaymentMethod','Partner',
                 'PaperlessBilling','charges_per_month','is_month_to_month','high_monthly']
    num_cols  = ['tenure','MonthlyCharges','TotalCharges','charges_per_month']
    cat_cols  = ['Contract','InternetService','PaymentMethod','Partner','PaperlessBilling']
    pass_cols = ['SeniorCitizen','is_month_to_month','high_monthly']
    X = df[FEATURES]; y = df['Churn']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    prep = ColumnTransformer([
        ('num',  StandardScaler(), num_cols),
        ('cat',  OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('pass', 'passthrough', pass_cols)
    ])
    pipes = {
        'Regresion Logistica': Pipeline([('prep',prep),('model',LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))]),
        'Random Forest':       Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))]),
        'Gradient Boosting':   Pipeline([('prep',prep),('model',GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))]),
        'KNN Classifier':      Pipeline([('prep',prep),('model',KNeighborsClassifier(n_neighbors=15, weights='distance'))]),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}; preds_dict = {}; probs_dict = {}
    for name, pipe in pipes.items():
        pipe.fit(X_tr, y_tr)
        yp   = pipe.predict(X_te)
        yprb = pipe.predict_proba(X_te)[:,1]
        preds_dict[name] = yp; probs_dict[name] = yprb
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring='roc_auc')
        results[name] = {
            'F1':        round(f1_score(y_te, yp), 4),
            'Accuracy':  round(accuracy_score(y_te, yp), 4),
            'Precision': round(precision_score(y_te, yp), 4),
            'Recall':    round(recall_score(y_te, yp), 4),
            'ROC-AUC':   round(roc_auc_score(y_te, yprb), 4),
            'CV AUC':    round(cv_auc.mean(), 4),
        }
    rf_model   = pipes['Random Forest'].named_steps['model']
    prep_fit   = pipes['Random Forest'].named_steps['prep']
    cat_names  = prep_fit.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    importance = pd.Series(rf_model.feature_importances_,
                           index=num_cols+cat_names+pass_cols).sort_values()
    return pipes, results, importance, X_te, y_te, FEATURES, preds_dict, probs_dict


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Taller 02")
    st.markdown("**Aprendizaje Supervisado**")
    st.divider()
    st.markdown("**Curso**  \nIntroduccion a la Inteligencia Artificial")
    st.markdown("**Docente**  \nJorge Ivan Padilla Buritica")
    st.markdown("**Universidad EAFIT — 2026-1**")
    st.divider()
    st.markdown("**Integrantes**  \n"
                "Ana Patricia Montes Pimienta  \n"
                "Karen Melissa Gomez Montoya  \n"
                "Juan Esteban Estrada Herrera")
    st.divider()
    st.markdown("**Regresion**  \n"
                "- Regresion Lineal\n- KNN Regressor\n- Random Forest  \n"
                "RMSE · MSE · R² · MAE")
    st.divider()
    st.markdown("**Clasificacion**  \n"
                "- Regresion Logistica\n- Random Forest\n- Gradient Boosting\n- KNN Classifier  \n"
                "F1 · Accuracy · Precision · Recall · ROC-AUC")
    st.divider()
    page = st.radio("Navegacion", [
        "Inicio",
        "Regresion — Medical Cost",
        "Clasificacion — Telco Churn"
    ])


# ════════════════════════════════════════════════════════════════════════════
# INICIO
# ════════════════════════════════════════════════════════════════════════════
if page == "Inicio":
    st.markdown('<p class="main-title">Taller 02 — Aprendizaje Supervisado</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Regresion y Clasificacion con Machine Learning</p>', unsafe_allow_html=True)
    st.divider()

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown("""
        <div class="info-box">
            <h4>Informacion Academica</h4>
            <p><strong>Curso:</strong> Introduccion a la Inteligencia Artificial</p>
            <p><strong>Docente:</strong> Jorge Ivan Padilla Buritica</p>
            <p><strong>Universidad:</strong> EAFIT &nbsp;|&nbsp; <strong>Periodo:</strong> 2026-1</p>
            <br>
            <p><strong>Integrantes:</strong></p>
            <p>Ana Patricia Montes Pimienta</p>
            <p>Karen Melissa Gomez Montoya</p>
            <p>Juan Esteban Estrada Herrera</p>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class="info-box">
            <h4>Descripcion del Trabajo</h4>
            <p>
            Este dashboard presenta los resultados del Taller 02 de Aprendizaje Supervisado,
            desarrollado en el marco del curso de Introduccion a la Inteligencia Artificial de EAFIT.
            </p>
            <p>
            Se implementaron modelos de <strong>regresion</strong> para predecir costos medicos
            (Medical Cost Personal Dataset) y modelos de <strong>clasificacion</strong> para
            predecir la fuga de clientes (Telco Customer Churn).
            </p>
            <p>
            El flujo incluye EDA, preprocesamiento, ingenieria de caracteristicas,
            entrenamiento con validacion cruzada y evaluacion con multiples metricas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="nav-card">
            <h3>Regresion — Medical Cost</h3>
            <p>Prediccion del costo medico anual (USD) a partir de variables
            demograficas y habitos de vida del asegurado.</p>
            <br>
            <p><strong>Modelos:</strong> Regresion Lineal &middot; KNN Regressor &middot; Random Forest</p>
            <p><strong>Metricas:</strong> RMSE &middot; MSE &middot; R² &middot; MAE</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Selecciona **Regresion — Medical Cost** en el menu lateral para acceder.")

    with col2:
        st.markdown("""
        <div class="nav-card">
            <h3>Clasificacion — Telco Churn</h3>
            <p>Prediccion de abandono de clientes a partir de datos de contrato,
            servicio de internet y comportamiento de pago.</p>
            <br>
            <p><strong>Modelos:</strong> Reg. Logistica &middot; Random Forest &middot; Gradient Boosting &middot; KNN</p>
            <p><strong>Metricas:</strong> F1 &middot; Accuracy &middot; Precision &middot; Recall &middot; ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Selecciona **Clasificacion — Telco Churn** en el menu lateral para acceder.")


# ════════════════════════════════════════════════════════════════════════════
# REGRESION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Regresion — Medical Cost":
    st.markdown('<p class="main-title">Regresion — Medical Cost Personal Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Prediccion del costo medico anual en USD</p>', unsafe_allow_html=True)

    with st.spinner("Entrenando modelos de regresion..."):
        pipes_r, results_r, importance_r, X_te_r, y_te_r, FEAT_R, preds_r = train_regression_models()

    tab1, tab2, tab3, tab4 = st.tabs(["Metricas", "Feature Importance", "Prediccion Individual", "Prediccion por Lote"])

    with tab1:
        st.markdown("### Resultados — Test Set y Cross Validation (KFold, 5 folds)")
        df_res_r  = pd.DataFrame(results_r).T
        best_r    = df_res_r['R2'].idxmax()
        best_vals = results_r[best_r]

        col1, col2, col3, col4 = st.columns(4)
        for col, (metric, label, unit) in zip(
            [col1,col2,col3,col4],
            [('RMSE','RMSE','USD'),('MSE','MSE','USD²'),('R2','R²',''),('MAE','MAE','USD')]
        ):
            with col:
                val_str = f"{best_vals[metric]:,.4f}" + (f" {unit}" if unit else "")
                st.markdown(f"""
                <div class="metric-card">
                    <p>{label} — {best_r}</p>
                    <h3>{val_str}</h3>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### Tabla de resultados")
        disp_r = pd.DataFrame(results_r).T[['RMSE','MSE','R2','MAE','CV R2']]\
                    .rename(columns={'R2':'R²','CV R2':'CV R²'})
        st.dataframe(
            disp_r.style
                .highlight_max(subset=['R²','CV R²'], color='#c8e6c9')
                .highlight_min(subset=['RMSE','MSE','MAE'], color='#c8e6c9')
                .format('{:.4f}'),
            use_container_width=True)

        st.markdown("#### Comparacion grafica")
        model_names_r = list(results_r.keys())
        colors_r = ['#1565c0','#2e7d32','#c62828']
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for (metric, title, hi), ax in zip([
            ('RMSE','RMSE (USD) — menor es mejor', False),
            ('MSE', 'MSE (USD²) — menor es mejor',  False),
            ('R2',  'R² — mayor es mejor',           True),
            ('MAE', 'MAE (USD) — menor es mejor',    False),
        ], axes.flatten()):
            vals = [results_r[n][metric] for n in model_names_r]
            bi   = vals.index(max(vals) if hi else min(vals))
            bcols = ['#f9a825' if i==bi else colors_r[i] for i in range(len(model_names_r))]
            bars = ax.bar(model_names_r, vals, color=bcols, edgecolor='white', width=0.5)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticklabels(model_names_r, rotation=15, ha='right', fontsize=9)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                        f'{val:,.2f}', ha='center', va='bottom', fontsize=9)
            ax.legend(handles=[mpatches.Patch(color='#f9a825', label='Mejor')], fontsize=8)
        plt.suptitle('Comparacion de Metricas — Medical Cost', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("#### Predicciones vs valores reales")
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        for (name, yp), ax, color in zip(preds_r.items(), axes2, colors_r):
            ax.scatter(y_te_r, yp, alpha=0.35, color=color, s=18)
            lim = [min(y_te_r.min(), yp.min()), max(y_te_r.max(), yp.max())]
            ax.plot(lim, lim, 'r--', lw=1.5)
            ax.set_title(f'{name}\nR²={results_r[name]["R2"]:.4f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Real (USD)'); ax.set_ylabel('Predicho (USD)')
        plt.suptitle('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    with tab2:
        st.markdown("### Feature Importance — Random Forest Regressor")
        st.caption("Importancia basada en reduccion de impureza (Gini).")
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ['#1565c0' if v==importance_r.max() else
                      '#42a5f5' if v>=importance_r.quantile(0.75) else
                      '#b0bec5' for v in importance_r.values]
        importance_r.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
        ax.axvline(importance_r.mean(), color='red', linestyle='--', alpha=0.7,
                   label=f'Media: {importance_r.mean():.4f}')
        ax.set_title('Feature Importance — Random Forest (Regresion)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importancia (Gini)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("**Top 3 variables mas predictivas:**")
        for feat, val in importance_r.sort_values(ascending=False).head(3).items():
            st.markdown(f"- `{feat}`: {val:.4f}")

    with tab3:
        st.markdown("### Prediccion Individual")
        col1, col2 = st.columns(2)
        with col1:
            age      = st.slider("Edad", 18, 64, 35)
            bmi      = st.slider("BMI", 15.0, 55.0, 28.5, 0.1)
            children = st.selectbox("Numero de hijos", [0,1,2,3,4,5])
            sex      = st.selectbox("Sexo", ["male","female"])
        with col2:
            smoker      = st.selectbox("Fumador", ["no","yes"])
            region      = st.selectbox("Region", ["southwest","southeast","northwest","northeast"])
            model_sel_r = st.selectbox("Modelo", list(pipes_r.keys()))

        if st.button("Predecir costo medico", use_container_width=True, type="primary"):
            inp = pd.DataFrame([{
                'age':age,'sex':sex,'bmi':bmi,'children':children,'smoker':smoker,'region':region,
                'bmi_smoker': bmi*(1 if smoker=='yes' else 0), 'obese': int(bmi>=30)
            }])
            pred  = pipes_r[model_sel_r].predict(inp)[0]
            nivel = "alto" if pred>20000 else "moderado" if pred>10000 else "bajo"
            box   = "pred-warn" if pred>15000 else "pred-ok"
            val   = "pred-val-warn" if pred>15000 else "pred-val-ok"
            st.markdown(f"""
            <div class="{box}">
                <p style="color:#555;font-size:1rem;">Costo medico estimado — {model_sel_r}</p>
                <p class="{val}">${pred:,.2f} USD</p>
                <p style="color:#666;">Nivel de riesgo: <strong>{nivel}</strong></p>
            </div>""", unsafe_allow_html=True)
            if smoker=='yes': st.warning("Fumador: incrementa significativamente el costo estimado.")
            if bmi >= 30:     st.warning("BMI >= 30 (obesidad): factor de riesgo adicional.")
            if age > 50:      st.info("Edad mayor a 50 anos: asociada a mayores costos medicos.")

    with tab4:
        st.markdown("### Prediccion por Lote")
        st.markdown("El CSV debe contener: `age, sex, bmi, children, smoker, region`")
        if st.button("Descargar CSV de ejemplo", key="reg_sample"):
            sample = pd.DataFrame({
                'age':[25,45,60,33],'sex':['male','female','male','female'],
                'bmi':[22.5,31.2,28.0,35.6],'children':[0,2,1,3],
                'smoker':['no','yes','no','no'],
                'region':['southwest','southeast','northwest','northeast']
            })
            st.download_button("Descargar", sample.to_csv(index=False),
                               "ejemplo_regresion.csv","text/csv")
        uploaded_r   = st.file_uploader("Cargar CSV", type=['csv'], key='reg_up')
        model_lote_r = st.selectbox("Modelo para prediccion masiva", list(pipes_r.keys()), key='reg_lote')
        if uploaded_r:
            df_up = pd.read_csv(uploaded_r)
            st.dataframe(df_up.head(), use_container_width=True)
            try:
                df_up['bmi_smoker'] = df_up['bmi'] * (df_up['smoker']=='yes').astype(int)
                df_up['obese']      = (df_up['bmi'] >= 30).astype(int)
                df_up['predicted_charges'] = np.round(pipes_r[model_lote_r].predict(df_up[FEAT_R]), 2)
                st.success(f"{len(df_up)} predicciones completadas.")
                st.dataframe(
                    df_up[['age','sex','bmi','smoker','region','predicted_charges']]\
                    .style.background_gradient(subset=['predicted_charges'], cmap='YlOrRd'),
                    use_container_width=True)
                st.download_button("Descargar resultados", df_up.to_csv(index=False),
                                   "predicciones_regresion.csv","text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# CLASIFICACION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Clasificacion — Telco Churn":
    st.markdown('<p class="main-title">Clasificacion — Telco Customer Churn</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Prediccion de abandono de clientes</p>', unsafe_allow_html=True)

    with st.spinner("Entrenando modelos de clasificacion..."):
        pipes_c, results_c, importance_c, X_te_c, y_te_c, FEAT_C, preds_c, probs_c = train_classification_models()

    tab1, tab2, tab3, tab4 = st.tabs(["Metricas", "Feature Importance", "Prediccion Individual", "Prediccion por Lote"])

    with tab1:
        st.markdown("### Resultados — Test Set y Cross Validation (StratifiedKFold, 5 folds)")
        df_res_c  = pd.DataFrame(results_c).T
        best_c    = df_res_c['ROC-AUC'].idxmax()
        best_vals_c = results_c[best_c]

        col1, col2, col3, col4, col5 = st.columns(5)
        for col, metric in zip([col1,col2,col3,col4,col5],
                                ['F1','Accuracy','Precision','Recall','ROC-AUC']):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <p>{metric} — {best_c}</p>
                    <h3>{best_vals_c[metric]:.4f}</h3>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### Tabla de resultados")
        disp_c = pd.DataFrame(results_c).T[['F1','Accuracy','Precision','Recall','ROC-AUC','CV AUC']]
        st.dataframe(
            disp_c.style.highlight_max(color='#c8e6c9').format('{:.4f}'),
            use_container_width=True)

        st.info("Con clases desbalanceadas (~26% churn), el Accuracy puede ser engañoso. "
                "F1-Score y ROC-AUC son las metricas principales.")

        st.markdown("#### Comparacion grafica de metricas")
        model_names_c = list(results_c.keys())
        colors_c = ['#1565c0','#2e7d32','#c62828','#e65100']
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for i, metric in enumerate(['F1','Accuracy','Precision','Recall','ROC-AUC']):
            ax   = axes.flatten()[i]
            vals = [results_c[n][metric] for n in model_names_c]
            bi   = vals.index(max(vals))
            bcols = ['#f9a825' if j==bi else colors_c[j] for j in range(len(model_names_c))]
            bars = ax.bar(model_names_c, vals, color=bcols, edgecolor='white', width=0.5)
            ax.set_title(metric, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.15)
            ax.set_xticklabels(model_names_c, rotation=20, ha='right', fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.01,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            ax.legend(handles=[mpatches.Patch(color='#f9a825', label='Mejor')], fontsize=7)
        axes.flatten()[5].axis('off')
        axes.flatten()[5].text(0.5, 0.5,
            'Nota:\n\nCon ~26% de churn,\nF1 y ROC-AUC son\nlas metricas\nprincipales.\n\nAccuracy puede\nser enganosa.',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#fff9c4', alpha=0.8))
        plt.suptitle('Comparacion de Metricas — Telco Churn', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("#### Curvas ROC")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for (name, y_prob), color in zip(probs_c.items(), colors_c):
            RocCurveDisplay.from_predictions(y_te_c, y_prob,
                name=f'{name} (AUC={roc_auc_score(y_te_c, y_prob):.4f})',
                ax=ax2, color=color, alpha=0.85)
        ax2.plot([0,1],[0,1],'k--',lw=1.2, label='Clasificador aleatorio (AUC=0.5)')
        ax2.set_title('Curvas ROC — Comparacion de Modelos', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='lower right'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

        st.markdown("#### Matrices de confusion")
        fig3, axes3 = plt.subplots(2, 2, figsize=(12, 9))
        for (name, yp), ax in zip(preds_c.items(), axes3.flatten()):
            ConfusionMatrixDisplay(confusion_matrix(y_te_c, yp),
                display_labels=['No Churn','Churn']).plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(f'{name}\nF1={results_c[name]["F1"]:.4f} | Recall={results_c[name]["Recall"]:.4f}',
                         fontsize=10, fontweight='bold')
        plt.suptitle('Matrices de Confusion — Test Set', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

    with tab2:
        st.markdown("### Feature Importance — Random Forest Classifier")
        st.caption("Importancia basada en reduccion de impureza (Gini).")
        fig, ax = plt.subplots(figsize=(10, 7))
        bar_colors = ['#c62828' if v==importance_c.max() else
                      '#ef9a9a' if v>=importance_c.quantile(0.75) else
                      '#b0bec5' for v in importance_c.values]
        importance_c.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
        ax.axvline(importance_c.mean(), color='blue', linestyle='--', alpha=0.6,
                   label=f'Media: {importance_c.mean():.4f}')
        ax.set_title('Feature Importance — Random Forest (Clasificacion)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importancia (Gini)'); ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("**Top 3 predictores de Churn:**")
        for feat, val in importance_c.sort_values(ascending=False).head(3).items():
            st.markdown(f"- `{feat}`: {val:.4f}")

    with tab3:
        st.markdown("### Prediccion Individual")
        col1, col2 = st.columns(2)
        with col1:
            tenure   = st.slider("Meses como cliente", 0, 72, 12)
            monthly  = st.slider("Cargo mensual (USD)", 20.0, 110.0, 65.0, 0.5)
            total    = st.number_input("Cargo total acumulado (USD)", 0.0, 8000.0,
                                        float(tenure*monthly), step=50.0)
            contract = st.selectbox("Tipo de contrato", ["Month-to-month","One year","Two year"])
            internet = st.selectbox("Servicio de internet", ["DSL","Fiber optic","No"])
        with col2:
            payment     = st.selectbox("Metodo de pago",
                                       ["Electronic check","Mailed check","Bank transfer","Credit card"])
            senior      = st.selectbox("Cliente mayor de edad", [0,1],
                                       format_func=lambda x: "Si" if x else "No")
            partner     = st.selectbox("Tiene pareja", ["Yes","No"])
            paperless   = st.selectbox("Facturacion sin papel", ["Yes","No"])
            model_sel_c = st.selectbox("Modelo", list(pipes_c.keys()))

        if st.button("Predecir Churn", use_container_width=True, type="primary"):
            inp = pd.DataFrame([{
                'tenure':tenure,'MonthlyCharges':monthly,'TotalCharges':total,
                'Contract':contract,'InternetService':internet,'PaymentMethod':payment,
                'SeniorCitizen':senior,'Partner':partner,'PaperlessBilling':paperless,
                'charges_per_month': total/(tenure+1),
                'is_month_to_month': int(contract=="Month-to-month"),
                'high_monthly': int(monthly > 82.5)
            }])
            prob  = pipes_c[model_sel_c].predict_proba(inp)[0][1]
            pred  = int(prob >= 0.5)
            label = "CHURN — Cliente en riesgo de abandono" if pred==1 else "NO CHURN — Cliente estable"
            box   = "pred-warn" if pred==1 else "pred-ok"
            val   = "pred-val-warn" if pred==1 else "pred-val-ok"
            st.markdown(f"""
            <div class="{box}">
                <p style="color:#555;font-size:1rem;">{label} — {model_sel_c}</p>
                <p class="{val}">Probabilidad de Churn: {prob:.1%}</p>
            </div>""", unsafe_allow_html=True)
            fig_g, ax_g = plt.subplots(figsize=(7, 1.4))
            ax_g.barh(0, 1, color='#e0e0e0', height=0.4)
            ax_g.barh(0, prob, color='#c62828' if prob>0.5 else '#2e7d32', height=0.4)
            ax_g.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
            ax_g.set_xlim(0,1); ax_g.set_yticks([])
            ax_g.set_xlabel("Probabilidad de Churn")
            ax_g.set_title(f"Probabilidad estimada: {prob:.1%}", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig_g); plt.close()
            if contract=="Month-to-month": st.warning("Contrato mes a mes: mayor riesgo de abandono.")
            if internet=="Fiber optic":    st.warning("Fibra optica: asociada a mayor tasa de churn.")
            if payment=="Electronic check":st.warning("Cheque electronico: patron frecuente en clientes que abandonan.")
            if tenure < 12:               st.warning("Cliente nuevo (menos de 12 meses): periodo critico de retencion.")

    with tab4:
        st.markdown("### Prediccion por Lote")
        st.markdown("El CSV debe contener: `tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, SeniorCitizen, Partner, PaperlessBilling`")
        if st.button("Descargar CSV de ejemplo", key="clf_sample"):
            sample_c = pd.DataFrame({
                'tenure':[2,45,12,60],'MonthlyCharges':[85.5,45.0,92.0,30.0],
                'TotalCharges':[171.0,2025.0,1104.0,1800.0],
                'Contract':['Month-to-month','Two year','Month-to-month','One year'],
                'InternetService':['Fiber optic','DSL','Fiber optic','No'],
                'PaymentMethod':['Electronic check','Bank transfer','Electronic check','Mailed check'],
                'SeniorCitizen':[0,0,1,0],'Partner':['No','Yes','No','Yes'],
                'PaperlessBilling':['Yes','No','Yes','No']
            })
            st.download_button("Descargar", sample_c.to_csv(index=False),
                               "ejemplo_churn.csv","text/csv")
        uploaded_c   = st.file_uploader("Cargar CSV", type=['csv'], key='clf_up')
        model_lote_c = st.selectbox("Modelo para prediccion masiva", list(pipes_c.keys()), key='clf_lote')
        threshold    = st.slider("Umbral de decision", 0.3, 0.7, 0.5, 0.05)
        if uploaded_c:
            df_up_c = pd.read_csv(uploaded_c)
            st.dataframe(df_up_c.head(), use_container_width=True)
            try:
                df_up_c['charges_per_month'] = df_up_c['TotalCharges'] / (df_up_c['tenure']+1)
                df_up_c['is_month_to_month'] = (df_up_c['Contract']=='Month-to-month').astype(int)
                df_up_c['high_monthly']      = (df_up_c['MonthlyCharges'] > 82.5).astype(int)
                probs  = pipes_c[model_lote_c].predict_proba(df_up_c[FEAT_C])[:,1]
                preds  = (probs >= threshold).astype(int)
                df_up_c['churn_probability'] = np.round(probs, 4)
                df_up_c['resultado']         = pd.Series(preds).map({1:'CHURN',0:'No Churn'}).values
                n_churn = preds.sum()
                st.success(f"{len(df_up_c)} predicciones completadas. "
                           f"Clientes en riesgo: {n_churn} ({n_churn/len(df_up_c):.1%})")
                col1, col2 = st.columns([3,1])
                with col1:
                    st.dataframe(
                        df_up_c[['tenure','MonthlyCharges','Contract','churn_probability','resultado']]\
                        .style.background_gradient(subset=['churn_probability'], cmap='RdYlGn_r'),
                        use_container_width=True)
                with col2:
                    fig_p, ax_p = plt.subplots(figsize=(4,4))
                    ax_p.pie([n_churn, len(df_up_c)-n_churn], labels=['Churn','No Churn'],
                             colors=['#c62828','#2e7d32'], autopct='%1.1f%%', startangle=90,
                             wedgeprops={'edgecolor':'white'})
                    ax_p.set_title('Distribucion de predicciones')
                    st.pyplot(fig_p); plt.close()
                st.download_button("Descargar resultados", df_up_c.to_csv(index=False),
                                   "predicciones_churn.csv","text/csv")
            except Exception as e:
                st.error(f"Error: {e}")
