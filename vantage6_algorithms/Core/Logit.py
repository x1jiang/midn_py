import numpy as np
from scipy import linalg
from scipy.special import expit
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector,
    write_string, write_integer, write_number, get_wrapped_websocket
)    
# R Original:
# ```r
# Logit = function(X,y,offset=rep(0,ncol(X)),beta0=rep(0,ncol(X)),lam=1e-3,maxiter=100)
# {
#   p = ncol(X)
#   n = nrow(X)
#   
#   beta = beta0
#   iter = 0
#   while ( iter < maxiter )
#   {
#     iter = iter + 1
#     
#     xb = drop(X%*%beta)
#     pr = 1/(1+exp(-xb))
#     H = t(X)%*%(X*pr*(1-pr)) + diag(n*lam,p)
#     g = t(X)%*%(y-pr) + n*offset - n*lam*beta
#     Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) + sum(n*offset*beta) - n*lam*sum(beta^2)/2
#     dir = drop(chol2inv(chol(H))%*%g)
#     m = sum(dir*g)
#     
#     step = 1
#     while (TRUE)
#     {
#       nbeta = beta + step*dir
#       if ( max(abs(nbeta-beta)) < 1e-5 )
#         break
#       xb = drop(X%*%nbeta)
#       pr = 1/(1+exp(-xb))
#       nQ = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) + sum(n*offset*nbeta) - n*lam*sum(nbeta^2)/2
#       if ( nQ-Q > m*step/2 )
#         break
#       step = step / 2
#     }
#     
#     if ( max(abs(nbeta-beta)) < 1e-5 )
#       break
#     beta = nbeta
#   }
#   
#   list(beta=beta,H=H)
# }
# ```

def Logit(X: np.ndarray, y: np.ndarray, offset: Optional[np.ndarray] = None, 
         beta0: Optional[np.ndarray] = None, lam: float = 1e-3, maxiter: int = 100) -> Dict[str, Any]:
    """
    Python implementation of the R Logit function
    
    Parameters:
    -----------
    X : np.ndarray
        Input matrix
    y : np.ndarray
        Binary target vector
    offset : np.ndarray, optional
        Offset vector, defaults to zeros
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    lam : float
        Regularization parameter
    maxiter : int
        Maximum number of iterations
        
    Returns:
    --------
    Dict with keys:
        beta: coefficient vector
        H: Hessian matrix
    """
    p = X.shape[1]
    n = X.shape[0]
    
    if offset is None:
        offset = np.zeros(p)
    
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0.copy()
    
    iter_count = 0
    while iter_count < maxiter:
        iter_count += 1
        
        xb = X @ beta
        pr = 1 / (1 + np.exp(-xb))
        H = X.T @ (X * pr[:, np.newaxis] * (1 - pr[:, np.newaxis])) + np.diag([n * lam] * p)
        g = X.T @ (y - pr) + n * offset - n * lam * beta
        
        # Calculate Q - log likelihood
        Q = np.sum(y * xb)
        Q += np.sum(np.log(1 - pr[pr < 0.5] + 1e-10))  # Add small constant to avoid log(0)
        Q += np.sum(np.log(pr[pr >= 0.5] + 1e-10) - xb[pr >= 0.5])
        Q += np.sum(n * offset * beta)
        Q -= n * lam * np.sum(beta**2) / 2
        
        try:
            dir_vec = linalg.cho_solve((linalg.cholesky(H, lower=False), False), g)
        except np.linalg.LinAlgError:
            # Fall back to a more robust but slower method if Cholesky decomposition fails
            dir_vec = np.linalg.solve(H, g)
            
        m = np.sum(dir_vec * g)
        
        step = 1
        while True:
            nbeta = beta + step * dir_vec
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                break
                
            xb = X @ nbeta
            pr = 1 / (1 + np.exp(-xb))
            
            # Calculate new Q
            nQ = np.sum(y * xb)
            nQ += np.sum(np.log(1 - pr[pr < 0.5] + 1e-10))
            nQ += np.sum(np.log(pr[pr >= 0.5] + 1e-10) - xb[pr >= 0.5])
            nQ += np.sum(n * offset * nbeta)
            nQ -= n * lam * np.sum(nbeta**2) / 2
            
            if nQ - Q > m * step / 2:
                break
            step = step / 2
        
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            break
        beta = nbeta
    
    return {'beta': beta, 'H': H}

# R Original:
# ```r
# SILogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#
#   K = length(rcons)
#   
#   beta = beta0
#   iter = 0
#   while ( iter < maxiter )
#   {
#     iter = iter + 1
#     
#     xb = drop(X%*%beta)
#     pr = 1/(1+exp(-xb))
#     H = t(X)%*%(X*pr*(1-pr))
#     g = t(X)%*%(y-pr)
#     Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
#     
#     N = n
#     if ( K > 0 )
#       for ( k in 1:K )
#       {
#         writeBin("Information",wcons[[k]])
#         writeBin("logistic",wcons[[k]])
#         writeBin(as.integer(yidx),wcons[[k]])
#         writeBin(as.integer(1),wcons[[k]])
#         writeVec(as.numeric(beta),wcons[[k]])
#         N = N + readVec(rcons[[k]])
#         H = H + readMat(rcons[[k]])
#         g = g + readVec(rcons[[k]])
#         Q = Q + readVec(rcons[[k]])
#       }
#     
#     H = H + diag(N*lam,p)
#     g = g - N*lam*beta
#     Q = Q - N*lam*sum(beta^2)/2
#     dir = drop(chol2inv(chol(H))%*%g)
#     m = sum(dir*g)
#     
#     step = 1
#     while (TRUE)
#     {
#       nbeta = beta + step*dir
#       if ( max(abs(nbeta-beta)) < 1e-5 )
#         break
#       
#       xb = drop(X%*%nbeta)
#       pr = 1/(1+exp(-xb))
#       nQ = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
#       if ( K > 0 )
#         for ( k in 1:K )
#         {
#           writeBin("Information",wcons[[k]])
#           writeBin("logistic",wcons[[k]])
#           writeBin(as.integer(yidx),wcons[[k]])
#           writeBin(as.integer(2),wcons[[k]])
#           writeVec(as.numeric(nbeta),wcons[[k]])
#           readVec(rcons[[k]])
#           nQ = nQ + readVec(rcons[[k]])
#         }
#       nQ = nQ - N*lam*sum(nbeta^2)/2
#
#       if ( nQ-Q > m*step/2 )
#         break
#       step = step / 2
#     }
#     
#     if ( max(abs(nbeta-beta)) < 1e-5 )
#       break
#     beta = nbeta
#   }
#   
#   list(beta=beta,H=H)
# }
# ```

async def SILogitNet(D: np.ndarray, idx: List[int], j: int,
                     remote_websockets: Dict, site_locks: Dict,
                     site_ids: List[str],
                     lam: float = 1e-3, maxiter: int = 100) -> Dict[str, np.ndarray]:
    """
    Asynchronous Python implementation of the R SILogitNet function with WebSocket communication
    
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
    maxiter : int
        Maximum number of iterations
        
    Returns:
    --------
    Dict with keys:
        beta: coefficient vector
        H: Hessian matrix
    """
    p = D.shape[1] - 1
    X = np.delete(D[idx, :], j, axis=1)                          # HIGHLIGHT (0-based j)
    y = D[idx, j]
    n = X.shape[0]

    beta = np.zeros(p, dtype=np.float64)

    for _ in range(maxiter):
        print(f"SILogitNet Iteration {_+1}", flush=True)
        start_time = asyncio.get_event_loop().time()
        xb = X @ beta
        pr = expit(xb)

        H = X.T @ (X * (pr * (1.0 - pr))[:, None])
        g = X.T @ (y - pr)
        Q = float(y @ xb
                  + np.log(1 - pr[pr < 0.5]).sum()
                  + (np.log(pr[pr >= 0.5]) - xb[pr >= 0.5]).sum())


        # Aggregate from remotes (using 1-based j for network)
        N = n
        for site_id in list(site_ids):
            if site_id not in remote_websockets:
                continue
            websocket = remote_websockets[site_id]
            async with site_locks[site_id]:
                ws = get_wrapped_websocket(websocket, pre_accepted=True)
                # print(f"[CENTRAL][SILOGIT  Requesting info from {site_id} for column (0-based) {j} for data {beta}, time {asyncio.get_event_loop().time() - start_time:.4f}s", flush=True)
                await write_string("Information", ws)
                await write_string("logistic", ws)
                await write_integer(j + 1, ws)                   # HIGHLIGHT (send 1-based)
                await write_integer(1, ws)                       # HIGHLIGHT (mode=1)
                await write_vector(beta.astype(np.float64), ws)  # HIGHLIGHT

                N += int((await read_vector(ws))[0])
                H += await read_matrix(ws)
                g += await read_vector(ws)
                Q += float((await read_vector(ws))[0])
                # print(f"[CENTRAL][SILOGIT  After aggregation from {site_id}: N={N}, H.shape={H.shape}, g.shape={g.shape}, Q={Q}, time={asyncio.get_event_loop().time() - start_time:.4f}s", flush=True)
        # Regularization with N*lam (R parity)
        H = H + (N * lam) * np.eye(p)                            # HIGHLIGHT
        g = g - (N * lam) * beta                                 # HIGHLIGHT
        Q = Q - (N * lam) * float(beta @ beta) / 2.0             # HIGHLIGHT

        # Direction via Cholesky (upper; like R's chol/backsolve)
        cH = linalg.cholesky(H, lower=False)                     # HIGHLIGHT
        direction = linalg.cho_solve((cH, False), g)             # HIGHLIGHT
        mval = float(direction @ g)
        # Backtracking line search (R rule)
        step = 1.0
        while True:
            nbeta = beta + step * direction
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                break

            xb = X @ nbeta
            prn = expit(xb)
            nQ = float(y @ xb
                       + np.log(1 - prn[prn < 0.5]).sum()
                       + (np.log(prn[prn >= 0.5]) - xb[prn >= 0.5]).sum())

            # Ask remotes for updated Q at nbeta (mode=2)
            for site_id in list(site_ids):
                if site_id not in remote_websockets:
                    continue
                websocket = remote_websockets[site_id]
                async with site_locks[site_id]:
                    ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    await write_string("Information", ws)
                    await write_string("logistic", ws)
                    await write_integer(j + 1, ws)               # HIGHLIGHT
                    await write_integer(2, ws)                   # HIGHLIGHT (mode=2)
                    await write_vector(nbeta.astype(np.float64), ws)
                    dummy_n = await read_vector(ws)                    # dummy n
                    nQ += float((await read_vector(ws))[0])      # HIGHLIGHT

            nQ = nQ - (N * lam) * float(nbeta @ nbeta) / 2.0     # HIGHLIGHT

            if (nQ - Q) > (mval * step / 2.0):                   # HIGHLIGHT (line-search rule)
                break
            step *= 0.5
        # print(f"Iter={_+1}, Q={Q:.4f}, step={step:.4f}, max|dir|={np.max(np.abs(direction)):.6f}, time={asyncio.get_event_loop().time() - start_time:.4f}s", flush=True)
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            beta = nbeta
            break
        beta = nbeta

    return {"beta": beta, "H": H}


# R Original:
# ```r
# CSLLogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#   
#   K = length(rcons)
#   
#   
#   fit1 = Logit(X,y,beta0=beta0)
#   beta1 = fit1$beta
#
#   N = n
#   offset = rep(0,p)
#   K = length(rcons)
#   if ( K > 0 )
#     for ( k in 1:K )
#     {
#       writeBin("Information",wcons[[k]])
#       writeBin("logistic",wcons[[k]])
#       writeBin(as.integer(yidx),wcons[[k]])
#       writeVec(beta1,wcons[[k]])
#       N = N + readVec(rcons[[k]])
#       offset = offset + readVec(rcons[[k]])
#     }
#   
#   
#   fit = Logit(X,y,offset,beta0=beta1)
#   fit
# }
# ```

async def CSLLogitNet(D: np.ndarray, idx: List[int], j: int, 
                          remote_websockets: Dict, site_locks: Dict,
                          site_ids: List[str], 
                          lam: float = 1e-3, 
                          beta0: Optional[np.ndarray] = None, 
                          maxiter: int = 100) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R CSLLogitNet function with WebSocket communication
    
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
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    maxiter : int
        Maximum number of iterations

    Returns:
    --------
    Dict with keys:
        beta: coefficient vector
        H: Hessian matrix
        iter: number of iterations performed
    """

    p = D.shape[1] - 1
    n = len(idx)
    

    # Initialize beta to zeros if not provided
    if beta0 is None:
        beta = np.zeros(p)
    else:
        beta = beta0.copy()
    
    # Get local data
    X = np.delete(D[idx, :], j, axis=1)
    y = D[idx, j]
    
    # Perform local logistic regression first
    fit1 = Logit(X, y, beta0=beta)
    beta1 = fit1['beta']
    
    N = n
    offset = np.zeros(p)
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
        print(f"[CENTRAL][CSL-LOGIT  No connected sites remaining", flush=True)
    
    # Request information from remote sites
    for site_id in site_ids:
        if site_id in remote_websockets:
            websocket = remote_websockets[site_id]
            site_start_time = asyncio.get_event_loop().time()
            try:
                async with site_locks[site_id]:
                    wrapped_ws = get_wrapped_websocket(websocket, pre_accepted=True)
                    
                    
                    # Send each message separately to ensure proper handling
                    #await asyncio.sleep(0.1)
                    cmd_size = await write_string("Information", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    method_size = await write_string("logistic", wrapped_ws)
                    
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
                print(f"[CENTRAL][CSL-LOGIT  Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][CSL-LOGIT  Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    
    # Apply the offset to create the final fit
    fit = Logit(X, y, offset=offset, beta0=beta1)
    
    return fit


# R Original:
# ```r
# AVGMLogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
# {
#   p = ncol(D)-1
#   n = length(idx)
#   
#   X = matrix(D[idx,-yidx],n,p)
#   y = D[idx,yidx]
#   
#   fit = Logit(X,y,lam=lam,beta0=beta0)
#   
#   AVGM = list()
#   AVGM$beta = fit$beta*n
#   AVGM$vcov = chol2inv(chol(fit$H))*n^2
#   AVGM$n = n
#   
#   K = length(hosts)
#   for ( k in 1:K )
#   {
#     writeBin("Information",wcons[[k]])
#     writeBin("logistic",wcons[[k]])
#     writeBin(as.integer(yidx),wcons[[k]])
#     
#     beta = readVec(rcons[[k]])
#     H = readMat(rcons[[k]])
#     n = readVec(rcons[[k]])
#     
#     AVGM$beta = AVGM$beta + beta*n
#     AVGM$vcov = AVGM$vcov + chol2inv(chol(H))*n^2
#     AVGM$n = AVGM$n + n
#   }
#   AVGM$beta = AVGM$beta/AVGM$n
#   AVGM$vcov = AVGM$vcov/AVGM$n^2
#   
#   AVGM
# }
# ```

async def AVGMLogitNet(D: np.ndarray, idx: List[int], j: int, 
                          remote_websockets: Dict, site_locks: Dict,
                          site_ids: List[str], 
                          lam: float = 1e-3, 
                          beta0: Optional[np.ndarray] = None, 
                          maxiter: int = 100) -> Dict[str, Any]:
    """
    Asynchronous Python implementation of the R AVGMLogitNet function with WebSocket communication
    
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
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    maxiter : int
        Maximum number of iterations
        
    Returns:
    --------
    Dict with AVGMLogit network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    

    # Initialize beta to zeros if not provided
    if beta0 is None:
        beta0 = np.zeros(p)
    
    # Get local data
    X = np.delete(D[idx, :], j, axis=1)
    y = D[idx, j]
    
    # Perform local logistic regression first
    fit = Logit(X, y, lam=lam, beta0=beta0)
    
    # Initialize AVGM dictionary
    AVGM = {}
    
    try:
        AVGM['beta'] = fit['beta'] * n
        AVGM['vcov'] = linalg.cho_solve((linalg.cholesky(fit['H'], lower=False), False), np.eye(p)) * (n**2)
        AVGM['n'] = n
    except np.linalg.LinAlgError:
        # Fall back to a more robust method if Cholesky decomposition fails
        print(f"[CENTRAL][AVGM-LOGIT  Cholesky decomposition failed, using direct inversion", flush=True)
        
        AVGM['beta'] = fit['beta'] * n
        AVGM['vcov'] = np.linalg.inv(fit['H']) * (n**2)
        AVGM['n'] = n
    
    # Check if we have any connected sites
    active_site_ids = [site_id for site_id in site_ids if site_id in remote_websockets]
    if not active_site_ids and len(site_ids) > 0:
        print(f"[CENTRAL][AVGM-LOGIT  No connected sites remaining", flush=True)
    
    # Request information from remote sites
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
                    await write_string("logistic", wrapped_ws)
                    
                    #await asyncio.sleep(0.1)
                    await write_integer(j+1, wrapped_ws)
                    
                    # Read results from the remote site with timeout to avoid deadlocks
                    beta_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    H_remote = await asyncio.wait_for(read_matrix(wrapped_ws), timeout=30.0)
                    n_remote = await asyncio.wait_for(read_vector(wrapped_ws), timeout=30.0)
                    
                    n_val = n_remote[0]
                    
                    try:
                        vcov_remote = linalg.cho_solve((linalg.cholesky(H_remote, lower=False), False), np.eye(p))
                    except np.linalg.LinAlgError:
                        # Fall back to a more robust method if Cholesky decomposition fails
                        vcov_remote = np.linalg.inv(H_remote)
                    
                    # Accumulate results
                    AVGM['beta'] += beta_remote * n_val
                    AVGM['vcov'] += vcov_remote * (n_val**2)
                    AVGM['n'] += n_val
                    
            except Exception as e:
                print(f"[CENTRAL][AVGM-LOGIT  Error communicating with {site_id}: {type(e).__name__}: {str(e)}", flush=True)
                
                # Remove the problematic site from our active connections
                if site_id in remote_websockets:
                    print(f"[CENTRAL][AVGM-LOGIT  Removing {site_id} from active connections", flush=True)
                    del remote_websockets[site_id]
    
    # Normalize the results
    AVGM['beta'] = AVGM['beta'] / AVGM['n']
    AVGM['vcov'] = AVGM['vcov'] / (AVGM['n']**2)
    
    return AVGM

# R Original:
# ```r
# ImputeLogit = function(yidx,alpha,wcons)
# {
#   K = length(wcons)
#   for ( k in 1:K )
#   {
#     writeBin("Impute",wcons[[k]])
#     writeBin("logistic",wcons[[k]])
#     writeBin(as.integer(yidx),wcons[[k]])
#     writeBin(as.numeric(alpha),wcons[[k]])
#   }
# }
# ```

async def ImputeLogit(j: int, alpha: np.ndarray,
                      remote_websockets: Dict, site_locks: Dict,
                      site_ids: List[str]) -> None:
    """
    Asynchronous Python implementation of the R ImputeLogit function with WebSocket communication
    
    Parameters:
    -----------
    j : int
        Index of target variable
    alpha : float
        Alpha value (probability threshold)
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
        print(f"[CENTRAL][IMPUTE-LOGIT No connected sites remaining", flush=True)
    # print(f"[CENTRAL][IMPUTE-LOGIT  Imputing column (0-based) {j} with alpha={alpha}", flush=True)
    # Send imputation information to remote sites
    for k, site_id in enumerate(site_ids):                           # HIGHLIGHT
        if site_id not in remote_websockets:
            continue
        websocket = remote_websockets[site_id]
        async with site_locks[site_id]:
            ws = get_wrapped_websocket(websocket, pre_accepted=True)
            await write_string("Impute", ws)
            await write_string("logistic", ws)
            await write_integer(j, ws)                                # 1-based if caller passed it
            await write_vector(alpha.astype(np.float64), ws)
    
