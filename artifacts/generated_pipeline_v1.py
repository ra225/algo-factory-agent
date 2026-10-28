import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score


class GeneratedCapability:
    """
    自动复刻算法能力组件
    选定方案节点: ['CAP_PREP_ROBUST', 'CAP_FE_CHURN_VELOCITY', 'CAP_MODEL_GBDT', 'CAP_POST_THRESH']
    修复轮次: Attempt #2
    """

    def __init__(self):
        self.num_cols = ["tenure", "monthly_charges", "total_charges", "support_calls",
                         "charge_velocity", "support_density"]
        self.cat_cols = ["contract_type", "payment_method"]
        self.preprocessor = None
        self.model = None
        self.best_threshold = 0.5

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()
        tenure_safe = X["tenure"].fillna(1.0).clip(lower=1.0)
        monthly_safe = X["monthly_charges"].fillna(50.0)
        total_safe = X["total_charges"].fillna(tenure_safe * monthly_safe)
        X["charge_velocity"] = total_safe / (tenure_safe * monthly_safe + 1.0)
        X["support_density"] = X["support_calls"].fillna(0.0) / tenure_safe
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_feat = self._engineer_features(X)
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, self.num_cols),
            ("cat", cat_pipe, self.cat_cols)
        ])
        X_trans = self.preprocessor.fit_transform(X_feat)
        self.model = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.05, class_weight='balanced', random_state=42)
        self.model.fit(X_trans, y)

        train_probs = self.model.predict_proba(X_trans)[:, 1]
        best_f1, best_th = 0.0, 0.5
        for th in np.linspace(0.30, 0.70, 17):
            score = f1_score(y, (train_probs >= th).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_th = score, float(th)
        self.best_threshold = best_th
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_feat = self._engineer_features(X)
        X_trans = self.preprocessor.transform(X_feat)
        return self.model.predict_proba(X_trans)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.best_threshold).astype(int)
