"""
sklearn_free.py - pure-NumPy reimplementations of the surrogate models used in the
manuscript, for environments where scikit-learn cannot be installed.

Implements, matching turbine_surrogate_training_pipeline.py:
  - StandardScaler
  - PolynomialFeatures(degree=2) + Ridge  (closed form)
  - ExtraTreesRegressor (Extremely Randomized Trees, bootstrap=False)
  - epsilon-SVR with RBF kernel (SMO solver)
  - KFold cross_val_predict, metrics

These reproduce the published modelling methodology; numerical values may differ
marginally from a scikit-learn run because of independent RNG streams and solver
tolerances.
"""
import numpy as np

# ----------------------------- metrics --------------------------------
def r2_score(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot
def mae(y, p): return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))

# --------------------------- preprocessing ----------------------------
class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, float); self.mean_ = X.mean(0); self.scale_ = X.std(0); self.scale_[self.scale_ == 0] = 1.0
        return self
    def transform(self, X): return (np.asarray(X, float) - self.mean_) / self.scale_
    def fit_transform(self, X): return self.fit(X).transform(X)

def poly2(X):
    """Degree-2 polynomial features (interactions + squares), no bias - matches
    sklearn PolynomialFeatures(degree=2, include_bias=False)."""
    X = np.asarray(X, float); n, d = X.shape
    cols = [X[:, j] for j in range(d)]                      # linear
    for i in range(d):                                      # squares & interactions
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

# ------------------------------ Ridge ---------------------------------
class PolyRidge:
    """StandardScaler -> PolynomialFeatures(2) -> Ridge(alpha). Intercept not penalised."""
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = poly2(self.sc.transform(X))
        self.zmean = Z.mean(0); self.ymean = float(np.mean(y))
        Zc = Z - self.zmean; yc = np.asarray(y, float) - self.ymean
        p = Zc.shape[1]
        self.w = np.linalg.solve(Zc.T @ Zc + self.alpha * np.eye(p), Zc.T @ yc)
        return self
    def predict(self, X):
        Z = poly2(self.sc.transform(X)) - self.zmean
        return Z @ self.w + self.ymean

# ------------------------- Extra Trees --------------------------------
class _ExtraTree:
    def __init__(self, min_samples_leaf=2, max_depth=20, rng=None):
        self.msl = min_samples_leaf; self.max_depth = max_depth; self.rng = rng
    def fit(self, X, y):
        self.tree = self._build(X, y, 0); return self
    def _build(self, X, y, depth):
        n = len(y)
        if n <= self.msl or depth >= self.max_depth or np.all(y == y[0]):
            return ("leaf", float(np.mean(y)))
        d = X.shape[1]
        best = None
        # Extra-Trees: one random threshold per feature, choose best variance reduction
        for f in range(d):
            lo, hi = X[:, f].min(), X[:, f].max()
            if hi <= lo:
                continue
            thr = self.rng.uniform(lo, hi)
            mask = X[:, f] <= thr
            nl, nr = mask.sum(), (~mask).sum()
            if nl < self.msl or nr < self.msl:
                continue
            var = (nl * np.var(y[mask]) + nr * np.var(y[~mask])) / n
            if best is None or var < best[0]:
                best = (var, f, thr, mask)
        if best is None:
            return ("leaf", float(np.mean(y)))
        _, f, thr, mask = best
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return ("node", f, thr, left, right)
    def _pred1(self, x, node):
        while node[0] == "node":
            _, f, thr, left, right = node
            node = left if x[f] <= thr else right
        return node[1]
    def predict(self, X): return np.array([self._pred1(x, self.tree) for x in X])

class ExtraTrees:
    def __init__(self, n_estimators=500, min_samples_leaf=2, random_state=2026):
        self.n = n_estimators; self.msl = min_samples_leaf; self.rs = random_state
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        rng = np.random.default_rng(self.rs)
        self.trees = []
        for _ in range(self.n):
            t = _ExtraTree(self.msl, rng=np.random.default_rng(rng.integers(1 << 31)))
            self.trees.append(t.fit(X, y))
        return self
    def predict(self, X):
        X = np.asarray(X, float)
        return np.mean([t.predict(X) for t in self.trees], axis=0)

# ------------------------------ SVR -----------------------------------
class SVR_RBF:
    """epsilon-SVR with RBF kernel, solved by a simple SMO. gamma='scale'
    (1/(n_features*Var(X))) on standardised inputs, matching sklearn defaults."""
    def __init__(self, C=10.0, epsilon=0.04, tol=1e-3, max_passes=2000):
        self.C = C; self.eps = epsilon; self.tol = tol; self.max_passes = max_passes
    def _K(self, A, B):
        a2 = np.sum(A * A, 1)[:, None]; b2 = np.sum(B * B, 1)[None, :]
        return np.exp(-self.gamma * (a2 + b2 - 2 * A @ B.T))
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        self.gamma = 1.0 / (Z.shape[1] * Z.var()) if Z.var() > 0 else 1.0
        self.Xtr = Z; y = np.asarray(y, float); n = len(y)
        K = self._K(Z, Z)
        beta = np.zeros(n)            # beta = alpha - alpha*  in [-C, C]
        Kb = np.zeros(n)             # K @ beta
        C, eps = self.C, self.eps
        # SMO with most-violating-pair selection on the smoothed (eps-insensitive) dual.
        for _ in range(self.max_passes):
            E = Kb - y               # residual = prediction (no bias) - target
            # subgradient of eps|beta| absorbed via shifted residual r
            r = E + eps * np.sign(beta)
            r[beta == 0] = E[beta == 0]      # at 0, eps-insensitive zone
            i = int(np.argmax(r)); j = int(np.argmin(r))
            if r[i] - r[j] < self.tol:
                break
            eta = K[i, i] + K[j, j] - 2 * K[i, j]
            if eta <= 1e-12:
                break
            bi, bj = beta[i], beta[j]
            d = -((E[i] - E[j]) + eps * (np.sign(bi) - np.sign(bj))) / eta
            ni = np.clip(bi + d, -C, C)
            nj = np.clip(bj - (ni - bi), -C, C)
            ni = np.clip(bi - (nj - bj), -C, C)
            di = ni - bi; dj = nj - bj
            if abs(di) < 1e-9 and abs(dj) < 1e-9:
                break
            beta[i] = ni; beta[j] = nj
            Kb += di * K[:, i] + dj * K[:, j]
        sv = (np.abs(beta) > 1e-6) & (np.abs(beta) < C - 1e-6)
        if sv.any():
            self.b = float(np.mean(y[sv] - Kb[sv] - eps * np.sign(beta[sv])))
        else:
            self.b = float(np.mean(y - Kb))
        self.beta = beta
        return self
    def predict(self, X):
        Z = self.sc.transform(X)
        return self._K(Z, self.Xtr) @ self.beta + self.b

# --------------------------- CV utilities -----------------------------
def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed); idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]

def cross_val_predict(make_model, X, y, k=5, seed=42):
    X = np.asarray(X, float); y = np.asarray(y, float); n = len(y)
    pred = np.zeros(n)
    for test in kfold_indices(n, k, seed):
        train = np.setdiff1d(np.arange(n), test)
        m = make_model().fit(X[train], y[train])
        pred[test] = m.predict(X[test])
    return pred
