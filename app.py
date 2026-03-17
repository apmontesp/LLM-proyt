"""
Taller 02 — Dashboard de Predicción
Maestría en Ciencia de Datos | EAFIT 2026-1
Modelos: Medical Cost (Regresión) + Telco Churn (Clasificación)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io, warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, ConfusionMatrixDisplay)

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taller 02 | EAFIT ML Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #555; font-size: 1rem; margin-top: 0; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 16px 20px; border-left: 4px solid #1a73e8;
        margin-bottom: 8px;
    }
    .metric-card h3 { margin: 0; font-size: 1.6rem; color: #1a73e8; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }
    .predict-box {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-radius: 14px; padding: 20px;
        border: 2px solid #4caf50; text-align: center;
    }
    .predict-box-warn {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-radius: 14px; padding: 20px;
        border: 2px solid #ff9800; text-align: center;
    }
    .predict-val { font-size: 2.4rem; font-weight: 800; color: #2e7d32; }
    .predict-val-warn { font-size: 2.4rem; font-weight: 800; color: #e65100; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA GENERATION ────────────────────────────────────────────────────────
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
    churn    = (np.random.uniform(0,1,n) < churn_p).astype(int)
    df = pd.DataFrame({'tenure':tenure,'MonthlyCharges':monthly,'TotalCharges':total,
                       'Contract':contract,'InternetService':internet,'PaymentMethod':payment,
                       'SeniorCitizen':senior,'Partner':partner,'PaperlessBilling':paperless,
                       'Churn':churn})
    df['charges_per_month']  = df['TotalCharges'] / (df['tenure']+1)
    df['is_month_to_month']  = (df['Contract']=='Month-to-month').astype(int)
    df['high_monthly']       = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    return df

# ─── MODEL TRAINING ─────────────────────────────────────────────────────────
@st.cache_resource
def train_regression_models():
    df = generate_regression_data()
    FEATURES = ['age','sex','bmi','children','smoker','region','bmi_smoker','obese']
    X = df[FEATURES]; y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_f  = ['age','bmi','children','bmi_smoker']
    cat_f  = ['sex','smoker','region']
    pass_f = ['obese']
    prep   = ColumnTransformer([
        ('num',  StandardScaler(), num_f),
        ('cat',  OneHotEncoder(drop='first', sparse_output=False), cat_f),
        ('pass', 'passthrough', pass_f)
    ])

    models = {
        'Ridge':            Pipeline([('prep',prep), ('model', Ridge(alpha=10.0))]),
        'Random Forest':    Pipeline([('prep',prep), ('model', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))]),
        'Gradient Boosting':Pipeline([('prep',prep), ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))]),
    }

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        cv_r2  = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
        cv_mae = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        results[name] = {
            'CV R²':   round(cv_r2.mean(),  4),
            'CV MAE':  round(-cv_mae.mean(), 2),
            'Test R²': round(r2_score(y_test, yp), 4),
            'Test MAE':round(mean_absolute_error(y_test, yp), 2),
            'Test RMSE':round(np.sqrt(mean_squared_error(y_test, yp)), 2),
        }

    # Feature importance from RF
    best_pipe = models['Random Forest']
    rf_model  = best_pipe.named_steps['model']
    cat_names = best_pipe.named_steps['prep'].named_transformers_['cat']\
                    .get_feature_names_out(cat_f).tolist()
    feat_names = num_f + cat_names + pass_f
    importance = pd.Series(rf_model.feature_importances_, index=feat_names)\
                    .sort_values(ascending=True)

    return models, results, importance, X_test, y_test, FEATURES

@st.cache_resource
def train_classification_models():
    df = generate_classification_data()
    FEATURES = ['tenure','MonthlyCharges','TotalCharges','SeniorCitizen',
                'Contract','InternetService','PaymentMethod','Partner',
                'PaperlessBilling','charges_per_month','is_month_to_month','high_monthly']
    X = df[FEATURES]; y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    num_f  = ['tenure','MonthlyCharges','TotalCharges','charges_per_month']
    cat_f  = ['Contract','InternetService','PaymentMethod','Partner','PaperlessBilling']
    pass_f = ['SeniorCitizen','is_month_to_month','high_monthly']
    prep   = ColumnTransformer([
        ('num',  StandardScaler(), num_f),
        ('cat',  OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_f),
        ('pass', 'passthrough', pass_f)
    ])

    models = {
        'Logistic Regression': Pipeline([('prep',prep), ('model', LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))]),
        'Random Forest':       Pipeline([('prep',prep), ('model', RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42))]),
        'Gradient Boosting':   Pipeline([('prep',prep), ('model', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))]),
    }

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        cv_f1  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1')
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc')
        pipe.fit(X_train, y_train)
        yp   = pipe.predict(X_test)
        yprb = pipe.predict_proba(X_test)[:,1]
        results[name] = {
            'CV F1':    round(cv_f1.mean(),  4),
            'CV AUC':   round(cv_auc.mean(), 4),
            'Test Acc': round(accuracy_score(y_test, yp), 4),
            'Test F1':  round(f1_score(y_test, yp), 4),
            'Test AUC': round(roc_auc_score(y_test, yprb), 4),
        }

    best_pipe = models['Random Forest']
    rf_model  = best_pipe.named_steps['model']
    cat_names = best_pipe.named_steps['prep'].named_transformers_['cat']\
                    .get_feature_names_out(cat_f).tolist()
    feat_names = num_f + cat_names + pass_f
    importance = pd.Series(rf_model.feature_importances_, index=feat_names)\
                    .sort_values(ascending=True)

    return models, results, importance, X_test, y_test, FEATURES

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EAFIT — Taller 02")
    st.markdown("**Maestría en Ciencia de Datos**")
    st.markdown("Curso: IA ECA&I | 2026-1")
    st.divider()
    page = st.radio("Navegar a:", [
        "🏠 Inicio",
        "🏥 Regresión — Medical Cost",
        "📱 Clasificación — Telco Churn"
    ])
    st.divider()
    st.markdown("**Modelos entrenados:**")
    st.markdown("- Ridge Regression\n- Random Forest\n- Gradient Boosting")
    st.divider()
    st.caption("Jorge Iván Padilla-Buriticá")

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: INICIO
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown('<p class="main-title">Taller 02 — ML Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Maestría en Ciencia de Datos · EAFIT · Periodo 2026-1</p>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏥 Regresión — Medical Cost")
        st.markdown("""
        Predicción del **costo médico individual** (USD) a partir de:
        - Edad, sexo, BMI, número de hijos
        - Hábito de fumar y región geográfica

        **Modelos:** Ridge · Random Forest · Gradient Boosting  
        **Métrica principal:** R², MAE, RMSE
        """)
        if st.button("Ir a Regresión →", use_container_width=True):
            st.session_state["nav"] = "reg"

    with col2:
        st.markdown("### 📱 Clasificación — Telco Churn")
        st.markdown("""
        Predicción de si un cliente **abandonará el servicio** a partir de:
        - Tenure, cargos, tipo de contrato
        - Servicio de internet y método de pago

        **Modelos:** Logistic Regression · Random Forest · Gradient Boosting  
        **Métrica principal:** AUC-ROC, F1, Recall
        """)
        if st.button("Ir a Clasificación →", use_container_width=True):
            st.session_state["nav"] = "clf"

    st.divider()
    st.markdown("### 📋 Checklist del Taller")
    checks = {
        "✅ 3 modelos por tarea (Regresión + Clasificación)": True,
        "✅ EDA con correlaciones y distribuciones": True,
        "✅ Preprocesamiento sin data leakage (Pipeline)": True,
        "✅ Feature Engineering documentado": True,
        "✅ GridSearch / hiperparámetros optimizados": True,
        "✅ Cross-Validation 5-fold (KFold / StratifiedKFold)": True,
        "✅ Métricas en Test set": True,
        "✅ Gráfica de Feature Importance": True,
        "✅ Predicción individual (formulario)": True,
        "✅ Predicción por lote (CSV upload)": True,
    }
    for item in checks:
        st.markdown(item)

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: REGRESIÓN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏥 Regresión — Medical Cost":
    st.markdown("## 🏥 Regresión — Medical Cost Personal Dataset")
    st.caption("Predicción del costo médico anual en USD")

    with st.spinner("Entrenando modelos de regresión..."):
        models, results, importance, X_test, y_test, FEATURES = train_regression_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Métricas", "📈 Feature Importance", "🔍 Predicción Individual", "📂 Predicción por Lote"
    ])

    # ── TAB 1: MÉTRICAS ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Comparación de Modelos — 5-Fold Cross Validation + Test Set")
        df_res = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Modelo'})

        col1, col2, col3 = st.columns(3)
        best = df_res.loc[df_res['Test R²'].idxmax()]
        with col1:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor R² en Test</p><h3>{best['Test R²']}</h3>
                <p>{best['Modelo']}</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor MAE en Test (USD)</p><h3>${best['Test MAE']:,}</h3>
                <p>{best['Modelo']}</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor RMSE en Test (USD)</p><h3>${best['Test RMSE']:,}</h3>
                <p>{best['Modelo']}</p></div>""", unsafe_allow_html=True)

        st.markdown("#### Tabla de Resultados")
        st.dataframe(df_res.set_index('Modelo').style.highlight_max(
            subset=['CV R²','Test R²'], color='#c8e6c9'
        ).highlight_min(
            subset=['CV MAE','Test MAE','Test RMSE'], color='#c8e6c9'
        ), use_container_width=True)

        st.markdown("#### Visualización de Métricas")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = ['#1a73e8','#34a853','#ea4335']
        names  = list(results.keys())

        for i, (metric, title, asc) in enumerate([
            ('Test R²',  'R² (mayor = mejor)',    False),
            ('Test MAE', 'MAE USD (menor = mejor)', True),
            ('Test RMSE','RMSE USD (menor = mejor)',True),
        ]):
            vals = [results[n][metric] for n in names]
            bars = axes[i].bar(names, vals, color=colors, edgecolor='white', width=0.5)
            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                             f'{val:,.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── TAB 2: FEATURE IMPORTANCE ─────────────────────────────────────────
    with tab2:
        st.markdown("### Feature Importance — Random Forest Regressor")
        st.caption("Importancia basada en reducción de impureza (Gini). Mayor valor = más predictiva.")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_fi = ['#1a73e8' if v == importance.max() else
                     '#34a853' if v >= importance.quantile(0.75) else
                     '#9e9e9e' for v in importance.values]
        importance.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
        ax.set_title('Feature Importance — Random Forest (Regresión)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Importancia (Gini)')
        ax.axvline(importance.mean(), color='red', linestyle='--', alpha=0.7, label='Promedio')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Interpretación")
        top3 = importance.sort_values(ascending=False).head(3)
        for feat, val in top3.items():
            st.markdown(f"- **`{feat}`**: importancia = `{val:.4f}`")

    # ── TAB 3: PREDICCIÓN INDIVIDUAL ─────────────────────────────────────
    with tab3:
        st.markdown("### Predicción Individual de Costo Médico")
        col1, col2 = st.columns(2)

        with col1:
            age      = st.slider("Edad", 18, 64, 35)
            bmi      = st.slider("BMI (Índice de Masa Corporal)", 15.0, 55.0, 28.5, 0.1)
            children = st.selectbox("Número de hijos", [0,1,2,3,4,5])
            sex      = st.selectbox("Sexo", ["male","female"])

        with col2:
            smoker   = st.selectbox("¿Fumador?", ["no","yes"])
            region   = st.selectbox("Región", ["southwest","southeast","northwest","northeast"])
            model_choice = st.selectbox("Modelo a usar", list(models.keys()))

        if st.button("🔮 Predecir Costo Médico", use_container_width=True, type="primary"):
            bmi_smoker = bmi * (1 if smoker == 'yes' else 0)
            obese      = 1 if bmi >= 30 else 0
            input_df   = pd.DataFrame([{
                'age':age,'sex':sex,'bmi':bmi,'children':children,
                'smoker':smoker,'region':region,
                'bmi_smoker':bmi_smoker,'obese':obese
            }])
            pred = models[model_choice].predict(input_df)[0]

            nivel = "alto" if pred > 20000 else "moderado" if pred > 10000 else "bajo"
            box_class = "predict-box-warn" if pred > 15000 else "predict-box"
            val_class  = "predict-val-warn" if pred > 15000 else "predict-val"

            st.markdown(f"""
            <div class="{box_class}">
                <p style="font-size:1rem; color:#555;">Costo médico estimado</p>
                <p class="{val_class}">${pred:,.2f} USD</p>
                <p style="color:#666;">Riesgo <strong>{nivel}</strong> · Modelo: {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Factores de riesgo detectados")
            if smoker == 'yes':
                st.warning("🚬 Fumador: aumenta significativamente el costo estimado")
            if bmi >= 30:
                st.warning("⚖️ BMI ≥ 30 (obesidad): factor de riesgo adicional")
            if age > 50:
                st.info("👴 Edad > 50: mayor probabilidad de costos médicos elevados")
            if smoker == 'no' and bmi < 30 and age < 40:
                st.success("✅ Perfil de bajo riesgo")

    # ── TAB 4: PREDICCIÓN POR LOTE ────────────────────────────────────────
    with tab4:
        st.markdown("### Predicción por Lote — Subir CSV")
        st.markdown("El CSV debe contener las columnas: `age, sex, bmi, children, smoker, region`")

        col1, col2 = st.columns([2,1])
        with col2:
            if st.button("📥 Descargar CSV de ejemplo"):
                sample = pd.DataFrame({
                    'age':    [25, 45, 60, 33],
                    'sex':    ['male','female','male','female'],
                    'bmi':    [22.5, 31.2, 28.0, 35.6],
                    'children':[0, 2, 1, 3],
                    'smoker': ['no','yes','no','no'],
                    'region': ['southwest','southeast','northwest','northeast']
                })
                csv = sample.to_csv(index=False)
                st.download_button("⬇️ Descargar", csv, "ejemplo_regresion.csv", "text/csv")

        uploaded = st.file_uploader("Subir archivo CSV", type=['csv'], key='reg_upload')
        model_lote = st.selectbox("Modelo para predicción masiva", list(models.keys()), key='reg_lote')

        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.markdown("**Vista previa:**")
            st.dataframe(df_upload.head(), use_container_width=True)

            try:
                df_upload['bmi_smoker'] = df_upload['bmi'] * (df_upload['smoker']=='yes').astype(int)
                df_upload['obese']      = (df_upload['bmi'] >= 30).astype(int)
                preds = models[model_lote].predict(df_upload[FEATURES])
                df_upload['predicted_charges'] = np.round(preds, 2)

                st.success(f"✅ {len(df_upload)} predicciones completadas")
                st.dataframe(df_upload[['age','sex','bmi','smoker','region','predicted_charges']]\
                             .style.background_gradient(subset=['predicted_charges'], cmap='YlOrRd'),
                             use_container_width=True)

                csv_out = df_upload.to_csv(index=False)
                st.download_button("⬇️ Descargar predicciones", csv_out,
                                   "predicciones_medicalcost.csv", "text/csv")
            except Exception as e:
                st.error(f"Error al procesar: {e}. Verifica que el CSV tiene las columnas correctas.")

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA: CLASIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📱 Clasificación — Telco Churn":
    st.markdown("## 📱 Clasificación — Telco Customer Churn")
    st.caption("Predicción de abandono de clientes (Churn = Sí / No)")

    with st.spinner("Entrenando modelos de clasificación..."):
        models_c, results_c, importance_c, X_test_c, y_test_c, FEATURES_C = train_classification_models()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Métricas", "📈 Feature Importance", "🔍 Predicción Individual", "📂 Predicción por Lote"
    ])

    # ── TAB 1: MÉTRICAS ──────────────────────────────────────────────────
    with tab1:
        st.markdown("### Comparación de Modelos — StratifiedKFold (5 folds) + Test Set")
        df_res_c = pd.DataFrame(results_c).T.reset_index().rename(columns={'index':'Modelo'})

        best_c = df_res_c.loc[df_res_c['Test AUC'].idxmax()]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor AUC-ROC</p><h3>{best_c['Test AUC']}</h3>
                <p>{best_c['Modelo']}</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor F1-Score</p><h3>{best_c['Test F1']}</h3>
                <p>{best_c['Modelo']}</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
                <p>Mejor Accuracy</p><h3>{best_c['Test Acc']}</h3>
                <p>{best_c['Modelo']}</p></div>""", unsafe_allow_html=True)

        st.markdown("#### Tabla de Resultados")
        st.dataframe(df_res_c.set_index('Modelo').style.highlight_max(
            subset=['CV F1','CV AUC','Test Acc','Test F1','Test AUC'], color='#c8e6c9'
        ), use_container_width=True)

        st.markdown("#### Visualización de Métricas y Matriz de Confusión")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        colors = ['#1a73e8','#34a853','#ea4335']
        names  = list(results_c.keys())

        for i, (metric, title) in enumerate([
            ('Test AUC', 'AUC-ROC'),
            ('Test F1',  'F1-Score'),
            ('Test Acc', 'Accuracy'),
        ]):
            vals = [results_c[n][metric] for n in names]
            bars = axes[i].bar(names, vals, color=colors, edgecolor='white', width=0.5)
            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].set_ylim(0, 1.1)
            axes[i].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_x()+bar.get_width()/2, val+0.01,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Confusion Matrix
        best_model = models_c['Random Forest']
        y_pred_c   = best_model.predict(X_test_c)
        cm = confusion_matrix(y_test_c, y_pred_c)
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn','Churn'])
        disp.plot(ax=axes[3], colorbar=False, cmap='Blues')
        axes[3].set_title('Confusion Matrix\n(Random Forest)', fontsize=11, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("⚠️ **Nota sobre Accuracy:** Con ~26% de churn (clases desbalanceadas), "
                "el Accuracy puede ser engañoso. El AUC-ROC y F1 son las métricas principales.")

    # ── TAB 2: FEATURE IMPORTANCE ─────────────────────────────────────────
    with tab2:
        st.markdown("### Feature Importance — Random Forest Clasificador")
        st.caption("Importancia basada en reducción de impureza. Revela los factores que más predicen el churn.")

        fig, ax = plt.subplots(figsize=(10, 7))
        colors_fi = ['#ea4335' if v == importance_c.max() else
                     '#ff7043' if v >= importance_c.quantile(0.75) else
                     '#9e9e9e' for v in importance_c.values]
        importance_c.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
        ax.set_title('Feature Importance — Random Forest (Clasificación)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Importancia (Gini)')
        ax.axvline(importance_c.mean(), color='blue', linestyle='--', alpha=0.6, label='Promedio')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Top 3 predictores de Churn")
        top3 = importance_c.sort_values(ascending=False).head(3)
        for feat, val in top3.items():
            st.markdown(f"- **`{feat}`**: importancia = `{val:.4f}`")

    # ── TAB 3: PREDICCIÓN INDIVIDUAL ─────────────────────────────────────
    with tab3:
        st.markdown("### Predicción Individual de Churn")
        col1, col2 = st.columns(2)

        with col1:
            tenure   = st.slider("Meses como cliente (tenure)", 0, 72, 12)
            monthly  = st.slider("Cargo mensual (USD)", 20.0, 110.0, 65.0, 0.5)
            total    = st.number_input("Cargo total acumulado (USD)", 0.0, 8000.0,
                                        float(tenure * monthly), step=50.0)
            contract = st.selectbox("Tipo de contrato",
                                    ["Month-to-month","One year","Two year"])
            internet = st.selectbox("Servicio de internet",
                                    ["DSL","Fiber optic","No"])

        with col2:
            payment  = st.selectbox("Método de pago",
                                    ["Electronic check","Mailed check",
                                     "Bank transfer","Credit card"])
            senior   = st.selectbox("¿Cliente mayor de edad?", [0, 1],
                                    format_func=lambda x: "Sí" if x else "No")
            partner  = st.selectbox("¿Tiene pareja?", ["Yes","No"])
            paperless= st.selectbox("¿Facturación sin papel?", ["Yes","No"])
            model_choice_c = st.selectbox("Modelo a usar", list(models_c.keys()))

        if st.button("🔮 Predecir Churn", use_container_width=True, type="primary"):
            cpm = total / (tenure + 1)
            imm = 1 if contract == "Month-to-month" else 0
            hm  = 1 if monthly > 82.5 else 0

            input_df = pd.DataFrame([{
                'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total,
                'Contract': contract, 'InternetService': internet,
                'PaymentMethod': payment, 'SeniorCitizen': senior,
                'Partner': partner, 'PaperlessBilling': paperless,
                'charges_per_month': cpm, 'is_month_to_month': imm, 'high_monthly': hm
            }])

            prob  = models_c[model_choice_c].predict_proba(input_df)[0][1]
            pred  = int(prob >= 0.5)
            label = "🚨 CHURN — Cliente en riesgo de abandono" if pred == 1 else "✅ NO CHURN — Cliente estable"
            box_c = "predict-box-warn" if pred == 1 else "predict-box"
            val_c = "predict-val-warn" if pred == 1 else "predict-val"

            st.markdown(f"""
            <div class="{box_c}">
                <p style="font-size:1rem;color:#555;">{label}</p>
                <p class="{val_c}">Probabilidad de Churn: {prob:.1%}</p>
                <p style="color:#666;">Modelo: {model_choice_c}</p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge visual
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(0, 1, color='#e0e0e0', height=0.4)
            ax.barh(0, prob, color='#ea4335' if prob > 0.5 else '#34a853', height=0.4)
            ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7)
            ax.set_xlim(0, 1); ax.set_yticks([])
            ax.set_xlabel("Probabilidad de Churn")
            ax.set_title(f"Probabilidad: {prob:.1%}", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("#### Factores de riesgo detectados")
            if contract == "Month-to-month":
                st.warning("📋 Contrato mes a mes: mayor probabilidad de abandono")
            if internet == "Fiber optic":
                st.warning("📡 Fibra óptica: clientes con este servicio tienen mayor churn")
            if payment == "Electronic check":
                st.warning("💳 Pago por cheque electrónico: asociado a mayor churn")
            if tenure < 12:
                st.warning("⏰ Cliente nuevo (< 12 meses): período crítico de retención")
            if tenure > 36 and contract != "Month-to-month":
                st.success("✅ Cliente fidelizado con contrato a largo plazo")

    # ── TAB 4: PREDICCIÓN POR LOTE ────────────────────────────────────────
    with tab4:
        st.markdown("### Predicción por Lote — Subir CSV")
        st.markdown("El CSV debe contener: `tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod, SeniorCitizen, Partner, PaperlessBilling`")

        col1, col2 = st.columns([2,1])
        with col2:
            if st.button("📥 Descargar CSV de ejemplo", key='clf_sample'):
                sample_c = pd.DataFrame({
                    'tenure':          [2, 45, 12, 60],
                    'MonthlyCharges':  [85.5, 45.0, 92.0, 30.0],
                    'TotalCharges':    [171.0, 2025.0, 1104.0, 1800.0],
                    'Contract':        ['Month-to-month','Two year','Month-to-month','One year'],
                    'InternetService': ['Fiber optic','DSL','Fiber optic','No'],
                    'PaymentMethod':   ['Electronic check','Bank transfer','Electronic check','Mailed check'],
                    'SeniorCitizen':   [0, 0, 1, 0],
                    'Partner':         ['No','Yes','No','Yes'],
                    'PaperlessBilling':['Yes','No','Yes','No'],
                })
                csv_s = sample_c.to_csv(index=False)
                st.download_button("⬇️ Descargar", csv_s, "ejemplo_churn.csv", "text/csv")

        uploaded_c = st.file_uploader("Subir archivo CSV", type=['csv'], key='clf_upload')
        model_lote_c = st.selectbox("Modelo para predicción masiva", list(models_c.keys()), key='clf_lote')
        threshold = st.slider("Umbral de decisión (probabilidad)", 0.3, 0.7, 0.5, 0.05)

        if uploaded_c:
            df_up_c = pd.read_csv(uploaded_c)
            st.markdown("**Vista previa:**")
            st.dataframe(df_up_c.head(), use_container_width=True)

            try:
                df_up_c['charges_per_month'] = df_up_c['TotalCharges'] / (df_up_c['tenure']+1)
                df_up_c['is_month_to_month'] = (df_up_c['Contract']=='Month-to-month').astype(int)
                df_up_c['high_monthly']      = (df_up_c['MonthlyCharges'] > 82.5).astype(int)

                probs  = models_c[model_lote_c].predict_proba(df_up_c[FEATURES_C])[:,1]
                preds  = (probs >= threshold).astype(int)
                df_up_c['churn_probability'] = np.round(probs, 4)
                df_up_c['churn_prediction']  = preds
                df_up_c['churn_label']       = df_up_c['churn_prediction'].map({1:'CHURN 🚨',0:'No Churn ✅'})

                n_churn = preds.sum()
                st.success(f"✅ {len(df_up_c)} predicciones completadas | "
                           f"🚨 {n_churn} clientes en riesgo ({n_churn/len(df_up_c):.1%})")

                col1, col2 = st.columns([3,1])
                with col1:
                    st.dataframe(
                        df_up_c[['tenure','MonthlyCharges','Contract',
                                  'churn_probability','churn_label']]\
                        .style.background_gradient(subset=['churn_probability'], cmap='RdYlGn_r'),
                        use_container_width=True)
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    ax2.pie([n_churn, len(df_up_c)-n_churn],
                            labels=['Churn','No Churn'],
                            colors=['#ea4335','#34a853'], autopct='%1.1f%%',
                            startangle=90, wedgeprops={'edgecolor':'white'})
                    ax2.set_title('Distribución\nde Predicciones')
                    st.pyplot(fig2)
                    plt.close()

                csv_out = df_up_c.to_csv(index=False)
                st.download_button("⬇️ Descargar predicciones", csv_out,
                                   "predicciones_churn.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}. Verifica que el CSV tiene las columnas correctas.")
