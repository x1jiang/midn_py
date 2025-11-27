import numpy as np
import asyncio
from scipy import linalg
from typing import List, Dict, Any, Optional, Union

# R Original:
# ```r
# LS = function(X,y,offset=rep(0,ncol(X)),lam=1e-3)
# {
#   p = ncol(X)
#   n = nrow(X)
#
#   XX = t(X)%*%X + diag(lam,p)
#   Xy = drop(t(X)%*%y)
#   yy = sum(y^2)
#   
#   cXX = chol(XX)
#   iXX = chol2inv(cXX)
#   
#   LS = list()
#   LS$beta = drop(iXX%*%(Xy-n*offset))
#   LS$n = n
#   e = drop(y - X%*%LS$beta)
#   LS$SSE = sum(e^2)
#   LS$cgram = cXX
#   
#   LS
# }
# ```

def LS(X: np.ndarray, y: np.ndarray, offset: Optional[np.ndarray] = None, lam: float = 1e-3) -> Dict[str, Any]:
    """
    Python implementation of the R LS function
    
    Parameters:
    -----------
    X : np.ndarray
        Input matrix
    y : np.ndarray
        Target vector
    offset : np.ndarray, optional
        Offset vector, defaults to zeros
    lam : float
        Regularization parameter
        
    Returns:
    --------
    Dict with keys:
        beta: coefficient vector
        n: number of samples
        SSE: sum of squared errors
        cgram: Cholesky decomposition of gram matrix
    """
    p = X.shape[1]
    n = X.shape[0]
    
    if offset is None:
        offset = np.zeros(p)
    
    XX = X.T @ X + np.diag([lam] * p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    cXX = linalg.cholesky(XX, lower=False)
    
    LS_result = {}
    LS_result['beta'] = linalg.cho_solve((cXX, False), (Xy - n * offset))  # HIGHLIGHT
    LS_result['n'] = n
    e = y - X @ LS_result['beta']
    LS_result['SSE'] = np.sum(e**2)
    LS_result['cgram'] = cXX
    
    return LS_result

# R Original:
# ```r
# SILSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#   XX = t(X)%*%X
#   Xy = drop(t(X)%*%y)
#   yy = sum(y^2)
#   
#   N = n
#   K = length(rcons)
#   if ( K > 0 )
#     for ( k in 1:K )
#     {
#       writeBin("Information",wcons[[k]])
#       writeBin("Gaussian",wcons[[k]])
#       writeBin(as.integer(yidx),wcons[[k]])
#       N = N + readVec(rcons[[k]])
#       XX = XX + readMat(rcons[[k]])
#       Xy = Xy + readVec(rcons[[k]])
#       yy = yy + readVec(rcons[[k]])
#     }
#   
#   cXX = chol(XX+diag(N*lam,p))
#   iXX = chol2inv(cXX)
#   
#   SILS = list()
#   SILS$beta = drop(iXX%*%Xy)
#   SILS$N = N
#   SILS$SSE = yy-crossprod(SILS$beta,2*Xy-XX%*%SILS$beta)
#   SILS$cgram = cXX
#   
#   SILS
# }
# ```

async def SILSNet(D: np.ndarray, idx: List[int], j: int, 
                 remote_websockets: Dict, site_locks: Dict,
                 site_ids: List[str], 
                 lam: float = 1e-3) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R SILSNet function with WebSocket communication
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    j : int
        Index of target variable
    remote_websockets : Dict
        Dictionary mapping site_id to websocket connections
    site_locks : Dict
        Dictionary mapping site_id to asyncio locks
    site_ids : List[str]
        List of remote site IDs to communicate with
    lam : float
        Regularization parameter
    debug_info : bool
        Whether to print debug information
    debug_counter : int
        Counter for debugging purposes
        
    Returns:
    --------
    Dict with SILS network results
    """

    p = D.shape[1] - 1
    n = len(idx)
    
    
    # Get local data
    X = np.delete(D[idx, :], j, axis=1)
    y = D[idx, j]
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    N = n
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
            print(f"[CENTRAL][SI-LS No connected sites remaining", flush=True)
    
    # Request information from remote sites
    from Core.transfer import get_wrapped_websocket, write_string, write_integer, read_vector, read_matrix
    
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            print(f"[CENTRAL][SI-LS  Requesting info from {site_id} for column (0-based) {j}", flush=True)
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)

                    #await asyncio.sleep(0.1)
                    await write_string("Information", wrapped_ws)

                    #await asyncio.sleep(0.1)
                    await write_string("Gaussian", wrapped_ws)

                    # HIGHLIGHT: send 1-based index to remote
                    j_r = j + 1
                    #await asyncio.sleep(0.1)
                    await write_integer(j_r, wrapped_ws)
                    
                    # Read results from the remote site with timeout to avoid deadlocks
                    N_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    N += N_remote[0]
                    
                    XX_remote = await asyncio.wait_for(read_matrix(wrapped_ws), timeout=30.0)
                    XX += XX_remote
                    
                    Xy_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    Xy += Xy_remote
                    
                    yy_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    yy += yy_remote[0]
                    
            except Exception as e:
                print(f"[CENTRAL][SI-LS Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][SI-LS Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    
    # Solve (XX + N*lam I) beta = Xy (numerically stable)
    cXX = linalg.cholesky(XX + (N * lam) * np.eye(p), lower=False)  
    beta = linalg.cho_solve((cXX, False), Xy)                       # (no explicit iXX)
    SILS = {
        "beta": beta,
        "N": N,
        "SSE": float(yy - beta @ (2.0 * Xy - (XX @ beta))),         # (parity with R)
        "cgram": cXX,
    }
    
    return SILS

# R Original:
# ```r
# ImputeLS = function(yidx,beta,sig,wcons)
# {
#   K = length(wcons)
#   for ( k in 1:K )
#   {
#     writeBin("Impute",wcons[[k]])
#     writeBin("Gaussian",wcons[[k]])
#     writeBin(as.integer(yidx),wcons[[k]])
#     writeBin(as.numeric(beta),wcons[[k]])
#     writeBin(as.numeric(sig),wcons[[k]])
#   }
# }
# ```

from Core.transfer import get_wrapped_websocket, write_string, write_integer, write_vector, write_number  # HIGHLIGHT

async def ImputeLS(j: int, beta: np.ndarray, sig: float,
                   remote_websockets: Dict, site_locks: Dict,
                   site_ids: List[str]) -> None:
    """
    Asynchronous Python implementation of the R ImputeLS function with WebSocket communication
    
    Parameters:
    -----------
    j : int
        Index of target variable
    beta : np.ndarray
        Coefficient vector
    sig : float
        Sigma value
    remote_websockets : Dict
        Dictionary mapping site_id to websocket connections
    site_locks : Dict
        Dictionary mapping site_id to asyncio locks
    site_ids : List[str]
        List of remote site IDs to communicate with
    """
  
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
        print(f"[CENTRAL][IMPUTE-LS No connected sites remaining", flush=True)
 
    
    # Check connections...
    for k, site_id in enumerate(site_ids):                           # HIGHLIGHT
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)

                    # Send command
                    #await asyncio.sleep(0.1)
                    await write_string("Impute", wrapped_ws)

                    #await asyncio.sleep(0.1)
                    await write_string("Gaussian", wrapped_ws)

                    # HIGHLIGHT: use 1-based index to match remote
                    j_r = j                                          # if caller already 1-based, keep; else j+1  # HIGHLIGHT
                    #await asyncio.sleep(0.1)
                    await write_integer(j_r, wrapped_ws)

                    # Payload
                    #await asyncio.sleep(0.1)
                    await write_vector(beta, wrapped_ws)
                    #await asyncio.sleep(0.1)
                    await write_vector(np.array([sig], dtype=np.float64), wrapped_ws)
                    
            except Exception as e:
                print(f"[CENTRAL][IMPUTE-LS Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][IMPUTE-LS Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    

# R Original:
# ```r
# CSLLSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#
#   XX = t(X)%*%X + diag(n*lam,p)
#   Xy = drop(t(X)%*%y)
#   yy = sum(y^2)
#   cXX = chol(XX)
#   iXX = chol2inv(cXX)
#   beta1 = drop(iXX%*%Xy)
#   
#   N = n
#   offset = rep(0,p)
#   K = length(rcons)
#   if ( K > 0 )
#     for ( k in 1:K )
#     {
#       writeBin("Information",wcons[[k]])
#       writeBin("Gaussian",wcons[[k]])
#       writeBin(as.integer(yidx),wcons[[k]])
#       writeVec(beta1,wcons[[k]])
#       N = N + readVec(rcons[[k]])
#       offset = offset + readVec(rcons[[k]])
#     }
#   
#   CSL = list()
#   CSL$beta = drop(iXX%*%(Xy-n*offset))
#   CSL$N = N
#   e = drop(y - X%*%CSL$beta)
#   CSL$SSE = sum(e^2)*N/n
#   gram = XX*N/n
#   CSL$cgram = chol(gram)
#   
#   CSL
# }
# ```

async def CSLLSNet(D: np.ndarray, idx: List[int], j: int, 
                       remote_websockets: Dict, site_locks: Dict,
                       site_ids: List[str], 
                       lam: float = 1e-3) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R CSLLSNet function with WebSocket communication
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    j : int
        Index of target variable
    remote_websockets : Dict
        Dictionary mapping site_id to websocket connections
    site_locks : Dict
        Dictionary mapping site_id to asyncio locks
    site_ids : List[str]
        List of remote site IDs to communicate with
    lam : float
        Regularization parameter
        
    Returns:
    --------
    Dict with CSLLS network results
    """

    p = D.shape[1] - 1
    n = len(idx)
    
    
    # Get local data
    X = np.delete(D[idx, :], j, axis=1)
    y = D[idx, j]
    
    # Compute local regression
    XX = X.T @ X + np.diag([n * lam] * p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    try:
        cXX = linalg.cholesky(XX, lower=False)
        iXX = linalg.cho_solve((cXX, False), np.eye(p))
    except np.linalg.LinAlgError:
        # Add more regularization if Cholesky decomposition fails
        print(f"[CENTRAL][CSL-LS Cholesky decomposition failed, adding more regularization", flush=True)
        XX = XX + np.eye(p) * 1e-4 * np.trace(XX) / p
        cXX = linalg.cholesky(XX, lower=False)
        iXX = linalg.cho_solve((cXX, False), np.eye(p))
    
    beta1 = iXX @ Xy
    
    N = n
    offset = np.zeros(p)
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
        print(f"[CENTRAL][CSL-LS No connected sites remaining", flush=True)
    
    # Request information from remote sites
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            site_start_time = asyncio.get_event_loop().time()
            try:
                async with site_locks[site_id]:
                    from Core.transfer import get_wrapped_websocket
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    
                    # Send instruction to remote site
                    from Core.transfer import write_string, write_integer, write_vector, read_vector
                    
                    # Send each message separately to ensure proper handling
                    #await asyncio.sleep(0.1)
                    await write_string("Information", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_string("Gaussian", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_integer(j+1, wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_vector(beta1, wrapped_ws)
                    
                    # Read results from the remote site with timeout to avoid deadlocks
                    n_site = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    N += int(n_site[0])  # Accumulate remote sample count
                    
                    offset_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    offset += offset_remote
                    
            except Exception as e:
                print(f"[CENTRAL][CSL-LS Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][CSL-LS Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    
    # Compute the final fit with the combined information
    # Instead of using LS function directly, we should follow the original R code implementation
    CSL = {}
    CSL['beta'] = iXX @ (Xy - n * offset)
    CSL['N'] = N
    e = y - X @ CSL['beta']
    CSL['SSE'] = np.sum(e**2) * N / n
    gram = XX * N / n
    CSL['cgram'] = linalg.cholesky(gram, lower=False)
    
    return CSL

# R Original:
# ```r
# AVGMLSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#   
#   XX = t(X)%*%X + diag(n*lam,p)
#   Xy = drop(t(X)%*%y)
#   yy = sum(y^2)
#   cXX = chol(XX)
#   iXX = chol2inv(cXX)
#   beta = drop(iXX%*%Xy)
#   SSE = yy + sum(beta*(XX%*%beta-2*Xy))
#   iFisher = iXX%*%XX%*%iXX
#
#   AVGM = list()
#   AVGM$beta = beta*n
#   AVGM$vcov = iFisher*n^2
#   AVGM$SSE = SSE
#   AVGM$N = n
#   
#   K = length(rcons)
#   for ( k in 1:K )
#   {
#     writeBin("Information",wcons[[k]])
#     writeBin("Gaussian",wcons[[k]])
#     writeBin(as.integer(yidx),wcons[[k]])
#     
#     beta = readVec(rcons[[k]])
#     iFisher = readMat(rcons[[k]])
#     SSE = readVec(rcons[[k]])
#     n = readVec(rcons[[k]])
#
#     AVGM$beta = AVGM$beta + beta*n
#     AVGM$vcov = AVGM$vcov + iFisher*n^2
#     AVGM$SSE = AVGM$SSE + SSE
#     AVGM$N = AVGM$N + n
#   }
#   AVGM$beta = AVGM$beta/AVGM$N
#   AVGM$vcov = AVGM$vcov/AVGM$N^2
#   
#   AVGM
# }
# ```

async def AVGMLSNet(D: np.ndarray, idx: List[int], j: int, 
                       remote_websockets: Dict, site_locks: Dict,
                       site_ids: List[str], 
                       lam: float = 1e-3 ) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R AVGMLSNet function with WebSocket communication
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    j : int
        Index of target variable
    remote_websockets : Dict
        Dictionary mapping site_id to websocket connections
    site_locks : Dict
        Dictionary mapping site_id to asyncio locks
    site_ids : List[str]
        List of remote site IDs to communicate with
    lam : float
        Regularization parameter
 
    Returns:
    --------
    Dict with AVGMLS network results
    """

    p = D.shape[1] - 1
    n = len(idx)
    
    
    # Get local data
    X = np.delete(D[idx, :], j, axis=1)
    y = D[idx, j]
    
    # Compute local regression
    XX = X.T @ X + np.diag([n * lam] * p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    try:
        cXX = linalg.cholesky(XX, lower=False)
        iXX = linalg.cho_solve((cXX, False), np.eye(p))
    except np.linalg.LinAlgError:
        # Add more regularization if Cholesky decomposition fails
        print(f"[CENTRAL][AVGM-LS Cholesky decomposition failed, adding more regularization", flush=True)
        XX = XX + np.eye(p) * 1e-4 * np.trace(XX) / p
        cXX = linalg.cholesky(XX, lower=False)
        iXX = linalg.cho_solve((cXX, False), np.eye(p))
    
    beta = iXX @ Xy
    SSE = yy + np.sum(beta * (XX @ beta - 2 * Xy))
    iFisher = iXX @ XX @ iXX
    
    # Initialize AVGM dictionary
    AVGM = {}
    AVGM['beta'] = beta * n
    AVGM['vcov'] = iFisher * (n**2)
    AVGM['SSE'] = SSE
    AVGM['N'] = n
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
        print(f"[CENTRAL][AVGM-LS No connected sites remaining", flush=True)
    
    # Request information from remote sites
    from Core.transfer import get_wrapped_websocket, write_string, write_integer, read_vector, read_matrix
    
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            site_start_time = asyncio.get_event_loop().time()
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)

                    
                    # Send each message separately to ensure proper handling
                    #await asyncio.sleep(0.1)
                    await write_string("Information", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_string("Gaussian", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_integer(j+1, wrapped_ws)
                    
                    # Read results from the remote site with timeout to avoid deadlocks
                    beta_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    iFisher_remote = await asyncio.wait_for(read_matrix(wrapped_ws), timeout=30.0)
                    SSE_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    n_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    
                    n_val = n_remote[0]
                    
                    # Accumulate results
                    AVGM['beta'] += beta_remote * n_val
                    AVGM['vcov'] += iFisher_remote * (n_val**2)
                    AVGM['SSE'] += SSE_remote[0]
                    AVGM['N'] += n_val
                    

            except Exception as e:
                print(f"[CENTRAL][AVGM-LS Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][AVGM-LS Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    
    # Normalize the results
    AVGM['beta'] = AVGM['beta'] / AVGM['N']
    AVGM['vcov'] = AVGM['vcov'] / (AVGM['N']**2)
    
 
    
    return AVGM
