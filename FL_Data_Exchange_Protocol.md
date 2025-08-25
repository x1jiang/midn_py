**SIMI **
Control (Central → Remote)

Gaussian

method: str → "Gaussian"

Logistic

method: str → "logistic"

Per-iteration:

Mode 1 (Hessian/gradient request)
mode: int → 1
beta: numpy.ndarray → shape (p,), dtype=float64

Mode 2 (objective-only request)
mode: int → 2
beta_candidate: numpy.ndarray → shape (p,), dtype=float64

Terminate
mode: int → 0

Data (Remote → Central)

Gaussian (one-shot)

n: float (use numpy.float64 or Python float)

XX: numpy.ndarray → shape (p,p), dtype=float64 (column-major / Fortran order when serialized)

Xy: numpy.ndarray → shape (p,), dtype=float64

yy: float (use numpy.float64 or Python float)

Logistic

Init (one-shot)
n: float (use numpy.float64 or Python float)

Mode 1 response (per request)
H: numpy.ndarray → shape (p,p), dtype=float64 (column-major / Fortran order when serialized)
g: numpy.ndarray → shape (p,), dtype=float64
Q: float (use numpy.float64 or Python float)

Mode 2 response (per request)
Q: float (use numpy.float64 or Python float)


**SIMICE**

Control (Central → Remote)

Initialize
message_type: str → "initialize"
target_column_indexes: List[int] (1-based)
is_binary: List[bool]

Request statistics
message_type: str → "request_info"  (* `request_statistics` → **request_info**)
target_column_index: int (1-based)
method: str → "gaussian" | "logistic"

Update imputations
message_type: str → "update_params"   (* `update_imputations` → **update_params**)
target_column_index: int (1-based)
method: str → "gaussian" | "logistic"
For gaussian:
beta: numpy.ndarray → shape (p,), dtype=float64
sigma: float (numpy.float64 or Python float)
For logistic:
alpha: numpy.ndarray → shape (p,), dtype=float64

Get final data
message_type: str → "get_final_data"

Data (Remote → Central)

Data prepared (one-shot)
message_type: str → "data_prepared"
n_observations: int
n_complete_cases: int
target_columns: List[int] (1-based)
is_binary: List[bool]
status: str → "initialized"

Statistics (response to Request statistics)

Gaussian
XTX: numpy.ndarray → shape (p,p), dtype=float64 (column-major / Fortran order when serialized)
XTy: numpy.ndarray → shape (p,), dtype=float64
yTy: float (numpy.float64 or Python float)
n: int

Logistic
H: numpy.ndarray → shape (p,p), dtype=float64 (column-major / Fortran order when serialized)
g: numpy.ndarray → shape (p,), dtype=float64
n: int
log_lik: float (numpy.float64 or Python float)

Update acknowledgement (optional)
message_type: str → "ack"   (* `iteration_complete` → **ack**)
status: str → "acknowledged" | "completed"

Final data (response to Get final data)
message_type: str → "final_data"
final_data: pandas.DataFrame
status: str → "completed"

**IMICE**
IMICECentral  (local, no exchange)
Inputs

D: numpy.ndarray → shape (n, p0), dtype=float64
Data matrix before intercept; continuous or binary targets are within these columns.

M: int → number of completed imputations to return

mvar: List[int] → column indices of variables to impute (use 0-based in Python; must be consistent)

method: List[str] → each item is "Gaussian" or "logistic", length == len(mvar)

iter: int → iterations between imputations (used after the first imputation)

iter0: int → iterations before the first imputation

Derived / internal state (local)

D_aug: numpy.ndarray → shape (n, p0+1), dtype=float64
D with a final intercept column of ones appended.

p: int → p0 + 1 (accounts for intercept)

miss: numpy.ndarray → shape (n, p), dtype=bool
Missingness mask over D_aug.

l: int → len(mvar)

j: int → current target column index from mvar (0-based)

idx1: numpy.ndarray → shape (nm_j,), dtype=int64
Row indices where D_aug[:, j] is missing.

idx0: numpy.ndarray → shape (nc_j,), dtype=int64
Row indices where D_aug[:, j] is observed.

Xidx: numpy.ndarray → shape (p-1,), dtype=int64
Predictor column indices = all columns except j.

midx: numpy.ndarray → shape (nmidx,), dtype=int64
Row indices where target j is missing (same as idx1 in the inner loop).

nmidx: int → number of missing rows for target j

Per-variable computation — Gaussian case (method[i] == "Gaussian")

fit.imp: mapping-like object (outputs of local least squares on observed rows)

n: int

SSE: float (numpy.float64 or Python float)

beta: numpy.ndarray → shape (p,), dtype=float64

cgram: numpy.ndarray → shape (p,p), dtype=float64
Cholesky factor of Gram matrix (e.g., chol(X'X)).

sig: float → sampled residual scale

alpha: numpy.ndarray → shape (p,), dtype=float64
Sampled regression coefficients used for imputation of missing rows.

D_aug[midx, j] (updated): numpy.ndarray → shape (nmidx,), dtype=float64
Imputed with linear predictor + Gaussian noise.

Per-variable computation — logistic case (method[i] == "logistic")

fit.imp: mapping-like object (outputs of local logistic fit on observed rows)

beta: numpy.ndarray → shape (p,), dtype=float64

H: numpy.ndarray → shape (p,p), dtype=float64
Approximate Hessian at beta.

cH: numpy.ndarray → shape (p,p), dtype=float64
Cholesky factor of H.

alpha: numpy.ndarray → shape (p,), dtype=float64
Sampled coefficient vector for imputation.

pr: numpy.ndarray → shape (nmidx,), dtype=float64
Bernoulli probabilities for missing rows (sigmoid(X_missing @ alpha)).

D_aug[midx, j] (updated): numpy.ndarray → shape (nmidx,), dtype=float64 with values in {0.0, 1.0}
Imputed via Bernoulli draws.

Outputs

imp: List[numpy.ndarray] → length M
Each element is a full imputed data matrix with intercept:

shape (n, p0+1), dtype=float64

All originally missing entries in columns mvar are imputed; final column is the intercept of ones.

**IMI**

IMICentral (local, no exchange)
Inputs

D: numpy.ndarray → shape (n, p+1), dtype=float64
(Includes an intercept column already; p = number of predictors excluding the imputed target.)

M: int → number of completed imputations to return

mvar: int → column index of the variable to impute (0-based in Python; must not be the intercept column)

method: str → "Gaussian" | "logistic"

Derived / internal state

n: int → number of rows in D

p: int → ncols(D) − 1 (predictor count used to impute mvar, including the intercept column among predictors)

miss: numpy.ndarray → shape (n,), dtype=bool
(Missing mask for the target column D[:, mvar])

nm: int → number of missing rows for target

nc: int → number of observed rows for target

X: numpy.ndarray → shape (nc, p), dtype=float64
(All columns except mvar; includes intercept column.)

y: numpy.ndarray → shape (nc,), dtype=float64
(Observed target values)

Model fit objects (from local fitting)

Gaussian (from IMICentralLS)

I.beta: numpy.ndarray → shape (p,), dtype=float64

I.vcov: numpy.ndarray → shape (p,p), dtype=float64 (covariance of beta)

I.SSE: float (numpy.float64 or Python float)

I.n: int

cvcov: numpy.ndarray → shape (p,p), dtype=float64 (Cholesky factor of I.vcov)

Logistic (from IMICentralLogit)

I.beta: numpy.ndarray → shape (p,), dtype=float64

I.vcov: numpy.ndarray → shape (p,p), dtype=float64 (inverse Hessian at final beta)

cvcov: numpy.ndarray → shape (p,p), dtype=float64 (Cholesky factor of I.vcov)

Per-imputation sampling and updates (for each m in 1..M)

Gaussian

sig: float (numpy.float64 or Python float)
(Sampled residual scale)

alpha: numpy.ndarray → shape (p,), dtype=float64
(Sampled coefficients: I.beta + sig * (cvcovᵀ @ Normal(0,I)))

D[miss, mvar] (updated): numpy.ndarray → shape (nm,), dtype=float64
(Imputed with D[miss, -mvar] @ alpha + Normal(0, sig))

Logistic

alpha: numpy.ndarray → shape (p,), dtype=float64
(Sampled coefficients: I.beta + (cvcovᵀ @ Normal(0,I)))

pr: numpy.ndarray → shape (nm,), dtype=float64
(Sigmoid of D[miss, -mvar] @ alpha)

D[miss, mvar] (updated): numpy.ndarray → shape (nm,), dtype=float64 with values in {0.0, 1.0}
(Bernoulli draws with probabilities pr)

Outputs

imp: List[numpy.ndarray] → length M
Each element is a full imputed data matrix (intercept column retained):
shape (n, p+1), dtype=float64


**CSLMICE**
Control (Central → Remote)

Initialize

inst: str → "Initialize"

mvar: List[int] (target column indices to impute)

Information (per target / per iteration)

inst: str → "Information"

method: str → "Gaussian" | "logistic"

yidx: int (target column index)

(after Remote sends n) betabar: numpy.ndarray → shape (p-1,), dtype=float64

Impute (broadcast parameters for site-side draws; no reply expected)

inst: str → "Impute"

method: str → "Gaussian"

yidx: int

beta: numpy.ndarray → shape (p-1,), dtype=float64

sig: float (numpy.float64 or Python float)

method: str → "logistic"

yidx: int

alpha: numpy.ndarray → shape (p-1,), dtype=float64

End

inst: str → "End"

Data (Remote → Central)

Information response

n: int (usable rows for this target at the site)

(after receiving betabar)

Gaussian: g: numpy.ndarray → shape (p-1,), dtype=float64

Logistic: g: numpy.ndarray → shape (p-1,), dtype=float64

(Initialize / Impute / End have no data back from Remote.)

Channel note

Two sockets per site: control (Central→Remote) and data (Remote→Central)

**CSLMI**
CSLMI — Python data types for on-wire messages (has exchange, no code)

Assumptions:

Indices below are 0-based for Python.

p = number of predictor columns used to impute the target (X = D[!miss, -mvar]), i.e., p = ncol(D) − 1.
(If an intercept exists in D, it’s part of X and thus counted in p.)

Control (Central → Remote)

Both methods (per request)

method: str → "Gaussian" | "logistic"

betabar: numpy.ndarray → shape (p,), dtype=float64
(Current coefficient iterate sent to the site.)

Data (Remote → Central)

Both methods (per request)

n: int
(Usable row count at the site for this target.)

g: numpy.ndarray → shape (p,), dtype=float64

Gaussian: site gradient contribution -Xᵀ(y − X·betabar)/n

Logistic: site gradient contribution Xᵀ(y − σ(X·betabar))/n

Direction & sequencing (per site, each round)

Central → Remote: method, betabar

Remote → Central: n, g

(No additional messages like “Impute” in CSLMI; Central performs imputation locally after aggregating all sites’ n and g.)

Channel note

Two sockets per site:

Control: Central → Remote (sends method, betabar)

Data: Remote → Central (returns n, g)


**AVGMMICE**

AVGMMICE — Python data types for on-wire messages (has exchange, no code)

Assumptions:

Indices are 0-based in Python terms.

p below means the number of columns in the site’s design X = DD[*, -yidx], i.e. p = (ncols(DD) − 1) because the target column yidx is excluded. (The intercept column is included inside X, so vectors/matrices are length/size p.)

Matrices are serialized column-major (Fortran order).

Control (Central → Remote)

Initialize

inst: str → "Initialize"

mvar: List[int] (target column indices to impute)

Information (per target / per iteration)

inst: str → "Information"

method: str → "Gaussian" | "logistic"

yidx: int (target column index)

Impute (broadcast site-side fill; no reply expected)

inst: str → "Impute"

method: str → "Gaussian"

yidx: int

beta: numpy.ndarray → shape (p,), dtype=float64

sig: float (numpy.float64 or Python float)

method: str → "logistic"

yidx: int

alpha: numpy.ndarray → shape (p,), dtype=float64

End

inst: str → "End"

Data (Remote → Central)

Information → Gaussian

beta: numpy.ndarray → shape (p,), dtype=float64

iFisher: numpy.ndarray → shape (p, p), dtype=float64 (column-major when serialized)

SSE: float (numpy.float64 or Python float)

n: int

Information → Logistic

beta: numpy.ndarray → shape (p,), dtype=float64

H: numpy.ndarray → shape (p, p), dtype=float64 (column-major when serialized)

n: int

Initialize / Impute / End

(no data returned)

Direction & sequencing (per site, each round)

Central → Remote: "Initialize", mvar → (no response)

Central → Remote: "Information", method, yidx

Remote → Central (Gaussian): beta, iFisher, SSE, n

Remote → Central (Logistic): beta, H, n

Central → Remote: "Impute", method, yidx, and (beta + sig) or (alpha) → (no response)

Central → Remote: "End" → (connection closes)

Channel note

Two sockets per site: control (Central→Remote) and data (Remote→Central).


**AVGMMI**
AVGMMI — Python data types for on-wire messages (has exchange, no code)

Assumptions:

Indices are 0-based (Python).

Let q be the number of predictors used for a target yidx: q = ncols(DD) − 1 (exclude the target column, include the intercept in X = DD[*, -yidx]).

Matrices are serialized column-major (Fortran order).

Control (Central → Remote)

Initialize

inst: str → "Initialize"

mvar: List[int] (target column indices to impute)

Information (per target / per iteration)

inst: str → "Information"

method: str → "Gaussian" | "logistic"

yidx: int (target column index)

Impute (broadcast parameters for site-side draws; no reply expected)

inst: str → "Impute"

method: str → "Gaussian"

yidx: int

beta: numpy.ndarray → shape (q,), dtype=float64

sig: float (numpy.float64 or Python float)

method: str → "logistic"

yidx: int

alpha: numpy.ndarray → shape (q,), dtype=float64

End

inst: str → "End"

Data (Remote → Central)

Information → Gaussian

beta: numpy.ndarray → shape (q,), dtype=float64

iFisher: numpy.ndarray → shape (q, q), dtype=float64 (column-major when serialized)

SSE: float (numpy.float64 or Python float)

n: int

Information → Logistic

beta: numpy.ndarray → shape (q,), dtype=float64

H: numpy.ndarray → shape (q, q), dtype=float64 (column-major when serialized)

n: int

Initialize / Impute / End

(no data returned)

Direction & sequencing (per site, each round)

Central → Remote: "Initialize", mvar → (no response)

Central → Remote: "Information", method, yidx

Remote → Central (Gaussian): beta, iFisher, SSE, n

Remote → Central (Logistic): beta, H, n

Central → Remote: "Impute", method, yidx, and (beta + sig) or (alpha) → (no response)

Central → Remote: "End" → (connection closes)

Channel note

Two sockets per site: control (Central→Remote) and data (Remote→Central).


**HDMI**
HDMI — Python data types for on-wire messages (has exchange, no code)

Assumptions

Indices below are 0-based (Python).

Let p be the number of columns in X = D[:, -mvar] (the target column is excluded; any intercept already in D remains inside X).

Matrices are serialized as float64, column-major (Fortran order).

Control (Central → Remote)

Single request per round

method: str → "Gaussian" | "logistic"

(No other control fields are sent. The remote already knows mvar from its own function arguments.)

Data (Remote → Central)
If method = "Gaussian"

beta: numpy.ndarray → shape (2*p + 1,), dtype=float64
Component layout (0-based indices):

beta_sel: beta[0 : p] (length p; selection model coefficients)

beta_out: beta[p : 2*p - 1] (length p−1; outcome model coefficients; note “exclusion” variable is omitted)

sigma: beta[2*p - 1] (scalar; log-σ in the code’s usage)

rho: beta[2*p] (scalar; dependence parameter)

vcov: numpy.ndarray → shape (2*p + 1, 2*p + 1), dtype=float64 (column-major when serialized)

SSE: Optional[float] (numpy.float64 or Python float, may be absent/empty; not used by central aggregation)

n: int (usable row count for this site)

If method = "logistic"

beta: numpy.ndarray → shape (2*p,), dtype=float64
Component layout (0-based indices):

beta_sel: beta[0 : p] (length p; selection model coefficients)

beta_out: beta[p : 2*p - 1] (length p−1; outcome model coefficients)

rho: beta[2*p - 1] (scalar; dependence parameter)

vcov: numpy.ndarray → shape (2*p, 2*p), dtype=float64 (column-major when serialized)

n: int (usable row count for this site)

Direction & sequencing (per site, one round)

Central → Remote: method

Remote → Central:

If "Gaussian": beta, vcov, SSE, n

If "logistic": beta, vcov, n

(Sockets then close. There’s no separate “Initialize” or “Impute” phase in HDMI; imputation is done centrally using the aggregated beta/vcov.)

---

### \u2B07 Unified-Schema (2025-08-25) compatibility notes
The Python dataclass layer (`data_protocol.py`) implements a *single* normalised wire schema that subsumes all message shapes described above.  Only naming and indexing were harmonised; the payload structure is unchanged.

* Indices are **0-based** (SIMICE examples showed 1-based).
* Methods are lower-case literals: `"gaussian"` | `"logistic"`.
* Message names
  * `request_statistics` → **request_info**
  * `update_imputations` → **update_params**
  * `iteration_complete` → **ack**
  * `get_final_data` (unchanged)
* The separate “Mode 1”, “Mode 2”, and “Terminate” verbs collapse into a single **iterate** message with an integer field `mode`:
  * 1 = H, g, (Q)
  * 2 = Q-only
  * 0 = terminate
* All vectors/matrices are `float64`; when flattened they use Fortran (column-major) order.
* Predictor dimension symbol is unified as **p** (some legacy text used *q*).
* HDMI beta blocks are transmitted as named fields: `beta_sel`, `beta_out`, `sigma`, `rho`, `vcov`.

The legacy protocol descriptions above remain verbatim for historical reference; implementers writing new code should follow the field names and conventions in `data_protocol.py`.
