# %% [markdown]
# # Modelo de predicción de pesos de peces a partir de medidas anatomicas

# %% [markdown]
# Preparar ambiente

# %%
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler

# %%
from pathlib import Path
import joblib
import time
import sklearn
import platform
import json
import os


# %%
class PipelineOptimizer:
    """Clase de contexto que utiliza el patrón estrategia"""

    def __init__(self, pipeline, X, y):
        self.pipeline = pipeline
        self.X = X
        self.y = y

    def optimize(self, strategy, param_definitions):
        """
        Parametros:
        ----------
            strategy: Estrategia de optimización
            param_definitions: Para GridSearch: diccionario de parámetros
                            Para Optuna: función que toma un trial y devuelve parámetros
        """
        return strategy.optimize(
            pipeline=self.pipeline,
            X=self.X,
            y=self.y,
            param_definitions=param_definitions,
        )


class OptimizationStrategy(ABC):
    """Clase base abstracta para estrategias de optimización"""

    @abstractmethod
    def optimize(self, pipeline, X, y, param_definitions):
        pass


class OptunaSearchStrategy(OptimizationStrategy):
    """Estrategia de Optuna"""

    def __init__(
        self,
        n_trials=100,
        cv=5,
        scoring="accuracy",
        direction="maximize",
        n_jobs=-1,
        sampler=TPESampler(),
    ):
        self.n_trials = n_trials
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.sampler = sampler
        self.n_jobs = n_jobs

    def optimize(self, pipeline, X, y, param_definitions):
        def objective(trial):
            # Obtener parámetros usando la función proporcionada por el usuario
            params = param_definitions(trial)
            pipeline_clone = clone(pipeline)
            pipeline_clone.set_params(**params)
            return cross_val_score(
                pipeline_clone,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()

        study = optuna.create_study(sampler=self.sampler, direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        return study


# Silenciar el output de optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
RANDOM_STATE = int(os.getenv("MODEL_RANDOM_STATE", 42))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./"))

current_dir = Path(__file__).parent.absolute()
data = current_dir / "Fish.csv"

# %% [markdown]
# Cargar datos y entrenar modelo

# %%
df = pd.read_csv(data)

# %%
# Separar características y variable objetivo
X = df.drop("Weight", axis=1)
y = df[["Weight"]]

# %%
# Dividir en conjunto de entrenamiento y prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# %%
# Identificar columnas numéricas y categóricas
numeric_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = X_train.select_dtypes(include=["category", "object"]).columns

# %%
# Escalar y
y_scaler = StandardScaler()
y_train_sc = y_scaler.fit_transform(y_train)
y_test_sc = y_scaler.transform(y_test)

# %% [markdown]
# Aplicamos preprocesamiento.
# Fijamos el grado del polinomio en 3.
# Grado 3 tiene sentido físico ya que corresponde a una medida de volumén.

# %%
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("poly", PolynomialFeatures(include_bias=False, degree=3)),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

# Combinar transformadores en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

Lassu = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(fit_intercept=True, max_iter=10000, random_state=RANDOM_STATE)),
    ]
)


# %% [markdown]
# Ajustamos modelos con optimización de hiperparametros


# %%
def LR_params(trial):
    return {
        "model__alpha": trial.suggest_float("model__alpha", 10**-2, 10**2),
    }


# %%
# Definir optimizador usando la clase y splits de cross validation
sampler = TPESampler(seed=RANDOM_STATE)
optimizerL = PipelineOptimizer(Lassu, X_train, y_train_sc)
cv = KFold(n_splits=5)
studyL = optimizerL.optimize(
    strategy=OptunaSearchStrategy(
        n_trials=200, cv=cv, scoring="r2", n_jobs=1, sampler=sampler
    ),
    param_definitions=LR_params,
)

# %%
# Entrenar los mejores hiperparametros usando todos los datos
best_params = studyL.best_params
best_L = clone(Lassu)
best_L.set_params(**best_params)
preprocessor = best_L.named_steps["preprocessor"]
preprocessor.set_output(transform="pandas")
best_L.fit(X_train, y_train_sc)

# %% [markdown]
# Guardamos el modelo

# %%
manifest = {
    "name": "Lasso-Prediccion-Peso-Fish",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "framework": "scikit-learn",
    "python_version": platform.python_version(),
    "pandas_version": pd.__version__,
    "sklearn_version": sklearn.__version__,
    "optuna_version": optuna.__version__,
    "random_state": RANDOM_STATE,
    "features": list(X.columns),
    "target": "Weight",
    "cv_metric": "r2",
    "cv_best_score": studyL.best_value,
    "cv_best_params": studyL.best_params,
    "test_metrics": best_L.score(X_test, y_test_sc),
}

# %%
ARTIFACTS_DIR.mkdir(exist_ok=True)
manifest_path = ARTIFACTS_DIR / "model_card.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

# %%
model_path = ARTIFACTS_DIR / "model.pkl"
joblib.dump(best_L, model_path)
