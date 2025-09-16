"""
Data protocol is now consolidated with job protocol in common.schema.protocol.
This module is kept for backward compatibility.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Literal
import numpy as np

# Import all classes from the consolidated protocol
from common.schema.protocol import (
    Method, Channel, MessageType,
    Envelope, GaussianInfo, LogisticInfo,
    IterateAskPayload, IterateResponsePayload,
    ParamUpdatePayload, HDMIParamsPayload
)

# Show deprecation warning
warnings.warn(
    "Importing from 'common.schema.data_protocol' is deprecated. "
    "Please update your imports to use 'common.schema.protocol' instead.",
    DeprecationWarning,
    stacklevel=2)

import warnings

# Import all classes from the consolidated protocol
from common.schema.protocol import (
    Method, Channel, MessageType,
    Envelope, GaussianInfo, LogisticInfo,
    IterateAskPayload, IterateResponsePayload,
    ParamUpdatePayload, HDMIParamsPayload
)

# Show deprecation warning
warnings.warn(
    "Importing from 'common.schema.data_protocol' is deprecated. "
    "Please update your imports to use 'common.schema.protocol' instead.",
    DeprecationWarning,
    stacklevel=2
)

# ------------------------------------------------------------------
# Legacy → normalized message-type mapping (for historical reference).
# These aliases are NOT used in the live schema but documented here
# so implementers migrating old code can see the correspondence at a
# glance.
#   request_statistics  → request_info
#   update_imputations  → update_params
#   iteration_complete  → ack
#   Mode 1 / Mode 2     → iterate with mode = 1 / 2
#   Terminate           → iterate with mode = 0
# ------------------------------------------------------------------

# ---- Message envelope (used everywhere a wire exists) ----
@dataclass
class Envelope:
    channel: Channel
    message_type: MessageType
    method: Optional[Method] = None
    yidx: Optional[int] = None            # ## CHANGE: unified 0-based index (SIMICE was 1-based)
    meta: Optional[Dict[str, Union[str, int, float]]] = None
    payload: Optional[Dict[str, object]] = None
    # Used by: SIMI, SIMICE, CSLMICE, CSLMI, AVGMMICE, AVGMMI, HDMI
    # (IMICE/IMI are local-only; no wire messages.)

# ---- Info/statistics payloads ----
@dataclass
class GaussianInfo:
    # One-shot sufficient stats OR curvature/grad per-iteration
    XTX: Optional[np.ndarray] = None   # (p,p) F-order; Used by: SIMI(one-shot), SIMICE, AVGMM*
    XTy: Optional[np.ndarray] = None   # (p,)   Used by: SIMI(one-shot), SIMICE
    yTy: Optional[float] = None        #        Used by: SIMI(one-shot), SIMICE
    H:   Optional[np.ndarray] = None   # (p,p)  Used by: SIMI(mode 1), AVGMM*
    g:   Optional[np.ndarray] = None   # (p,)   Used by: SIMI(mode 1), CSLMICE/CSLMI (grad rounds)
    Q:   Optional[float] = None        #        Used by: SIMI(mode 1/2 objective)
    n:   Optional[int] = None          #        Used by: SIMI/SIMICE/CSL*/AVGMM*/HDMI (as applicable)

@dataclass
class LogisticInfo:
    H:       Optional[np.ndarray] = None  # (p,p)  Used by: SIMI, SIMICE, AVGMM*
    g:       Optional[np.ndarray] = None  # (p,)   Used by: SIMI, CSLMICE/CSLMI (grad rounds)
    n:       Optional[int] = None         #        Used by: SIMI/SIMICE/CSL*/AVGMM*/HDMI
    log_lik: Optional[float] = None       #        Used by: SIMICE
    Q:       Optional[float] = None       #        Used by: SIMI (objective)

# ---- Iteration ask/response (covers SIMI modes + CSL* grad rounds) ----
@dataclass
class IterateAskPayload:
    mode: int                               # 1=H,g,(Q) | 2=Q-only | 0=terminate   ## CHANGE: unified mode flag
    beta: Optional[np.ndarray] = None       # (p,) Gaussian; Used by: SIMI
    beta_candidate: Optional[np.ndarray] = None  # (p,) SIMI mode 2
    alpha: Optional[np.ndarray] = None      # (p,) Logistic; Used by: SIMI
    betabar: Optional[np.ndarray] = None    # (p,) Used by: CSLMI/CSLMICE gradient rounds  ## CHANGE: added field for CSL*

@dataclass
class IterateResponsePayload:
    H:   Optional[np.ndarray] = None  # (p,p); Used by: SIMI mode 1
    g:   Optional[np.ndarray] = None  # (p,);  Used by: SIMI, CSLMI, CSLMICE (some variants emit p-1)  ## CHANGE: dim note
    Q:   Optional[float] = None       #       Used by: SIMI modes 1/2
    n:   Optional[int] = None         #       Used by: CSLMI/CSLMICE gradient rounds

# ---- Parameter update / imputation broadcast ----
@dataclass
class ParamUpdatePayload:
    beta:  Optional[np.ndarray] = None  # (p,) Gaussian; Used by: SIMICE update, CSLMICE impute, AVGMM*
    alpha: Optional[np.ndarray] = None  # (p,) Logistic; Used by: SIMICE update, CSLMICE impute, AVGMM*
    sigma: Optional[float] = None       # Gaussian scale; Used by: SIMICE/CSLMICE/AVGMM*   ## CHANGE: normalized name

# ---- HDMI packed parameter block (normalized) ----
@dataclass
class HDMIParamsPayload:
    beta_sel: Optional[np.ndarray] = None  # (p,)     selection model
    beta_out: Optional[np.ndarray] = None  # (p-1,)   outcome model (exclusion handled site-side)
    sigma:    Optional[float] = None       # Gaussian only (HDMI); upstream may be log-σ  ## CHANGE: normalized to σ
    rho:      Optional[float] = None
    vcov:     Optional[np.ndarray] = None  # full vcov (Fortran order)
    n:        Optional[int] = None
    # Used by: HDMI (single-round info responses)

# ---- Mapping remarks (which algorithm uses what) ----
# SIMI:
#   - Envelope(iterate/iter_response) with IterateAskPayload(mode, beta/alpha/beta_candidate) and IterateResponsePayload(H,g,Q[,n]).
#   - Optional one-shot GaussianInfo via request_info/info (XTX, XTy, yTy, n).
#
# SIMICE:
#   - Envelope(initialize/request_info/update_params/get_final_data/final_data).
#   - GaussianInfo / LogisticInfo for statistics; ParamUpdatePayload(beta/alpha/sigma) for updates.
#
# IMICE / IMI:
#   - Local-only (no Envelope on wire); included here for completeness of schema.
#
# CSLMICE:
#   - initialize → request_info(info.n) → iterate(betabar) → iter_response(n,g) → impute(beta/sigma | alpha) → finalize.
#
# CSLMI:
#   - repeated iterate(betabar) → iter_response(n,g); central performs imputation; no impute broadcast.
#
# AVGMMICE / AVGMMI:
#   - initialize → request_info(info: beta, iFisher/H, (SSE), n) using GaussianInfo/LogisticInfo fields → impute broadcast → finalize.
#
# HDMI:
#   - request_info(method) → info(HDMIParamsPayload); central imputes; no impute broadcast.

# ---- Global remarks ----
# ## CHANGE: All indices are 0-based (SIMICE originally 1-based).
# ## CHANGE: Methods unified to lowercase "gaussian"/"logistic".
# ## CHANGE: Single 'iterate' with numeric 'mode' covers SIMI Mode 1/2 and termination.
# ## CHANGE: Float64 everywhere; matrices serialized column-major (Fortran order).
# ## CHANGE: Predictor dimension unified as 'p' across protocols.
# ## CHANGE: HDMI concatenations are split into explicit fields (beta_sel, beta_out, sigma, rho).
