import numpy as np
import pandas as pd
from scipy.special import expit

def infer_simi(model: dict, df: pd.DataFrame, deterministic: bool = True) -> pd.DataFrame:
    """Impute missing values locally using returned SIMI parameters."""
    df = df.copy()
    col_idx = int(model.get("target_column_index", 1)) - 1
    feature_idx = model.get("feature_indices", [])
    beta = np.array(model.get("beta", []))
    method = model.get("method", "Gaussian").lower()
    X = df.iloc[:, feature_idx].values
    missing = df.iloc[:, col_idx].isna().to_numpy()

    if method == "gaussian":
        mu = X @ beta
        if deterministic:
            imputed = mu
        else:
            sigma2 = float(model.get("sigma2", 0.0))
            imputed = mu + np.random.normal(0, np.sqrt(max(sigma2, 1e-8)), size=len(mu))
    else:
        pr = expit(X @ beta)
        imputed = (pr >= 0.5).astype(float) if deterministic else np.random.binomial(1, pr)

    df.iloc[:, col_idx] = np.where(missing, imputed, df.iloc[:, col_idx])
    return df
