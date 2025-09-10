import numpy as np
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
    iXX = linalg.cho_solve((cXX, False), np.eye(p))
    
    LS_result = {}
    LS_result['beta'] = iXX @ (Xy - n * offset)
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

def SILSNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3, 
           rcons: Optional[List] = None, wcons: Optional[List] = None) -> Dict[str, Any]:
    """
    Python implementation of the R SILSNet function
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    yidx : int
        Index of target variable
    lam : float
        Regularization parameter
    rcons, wcons : communication channels for distributed processing
        
    Returns:
    --------
    Dict with SILS network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    XX = X.T @ X
    Xy = X.T @ y
    yy = np.sum(y**2)
    
    N = n
    K = 0 if rcons is None else len(rcons)
    
    if K > 0:
        # Using the communication functions from transfer.py
        import asyncio
        from Core.transfer import write_string, write_integer, read_vector, read_matrix
        
        # Need to run this in an async context
        async def communicate():
            nonlocal N, XX, Xy, yy
            for k in range(K):
                await write_string("Information", wcons[k])
                await write_string("Gaussian", wcons[k])
                await write_integer(yidx, wcons[k])
                
                # Read data from remote sites
                N_remote = await read_vector(rcons[k])
                N += N_remote[0]
                XX_remote = await read_matrix(rcons[k])
                XX += XX_remote
                Xy_remote = await read_vector(rcons[k])
                Xy += Xy_remote
                yy_remote = await read_vector(rcons[k])
                yy += yy_remote[0]
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())
    
    cXX = linalg.cholesky(XX + np.diag([N * lam] * p), lower=False)
    iXX = linalg.cho_solve((cXX, False), np.eye(p))
    
    SILS = {}
    SILS['beta'] = iXX @ Xy
    SILS['N'] = N
    SILS['SSE'] = yy - np.dot(SILS['beta'], 2 * Xy - XX @ SILS['beta'])
    SILS['cgram'] = cXX
    
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

def ImputeLS(yidx: int, beta: np.ndarray, sig: float, wcons: List) -> None:
    """
    Python implementation of the R ImputeLS function
    
    Parameters:
    -----------
    yidx : int
        Index of target variable
    beta : np.ndarray
        Coefficient vector
    sig : float
        Sigma value
    wcons : communication channels for distributed processing
    """
    K = 0 if wcons is None else len(wcons)
    
    if K > 0:
        # Using the communication functions from transfer.py
        import asyncio
        from Core.transfer import write_string, write_integer, write_vector
        
        # Need to run this in an async context
        async def communicate():
            for k in range(K):
                await write_string("Impute", wcons[k])
                await write_string("Gaussian", wcons[k])
                await write_integer(yidx, wcons[k])
                await write_vector(beta, wcons[k])
                await write_vector(np.array([sig]), wcons[k])
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())

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

def CSLLSNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3,
            rcons: Optional[List] = None, wcons: Optional[List] = None) -> Dict[str, Any]:
    """
    Python implementation of the R CSLLSNet function
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    yidx : int
        Index of target variable
    lam : float
        Regularization parameter
    rcons, wcons : communication channels for distributed processing
        
    Returns:
    --------
    Dict with CSLLS network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    
    XX = X.T @ X + np.diag([n * lam] * p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    cXX = linalg.cholesky(XX, lower=False)
    iXX = linalg.cho_solve((cXX, False), np.eye(p))
    beta1 = iXX @ Xy
    
    N = n
    offset = np.zeros(p)
    K = 0 if rcons is None else len(rcons)
    
    if K > 0:
        # Using the communication functions from transfer.py
        import asyncio
        from Core.transfer import write_string, write_integer, write_vector, read_vector
        
        # Need to run this in an async context
        async def communicate():
            nonlocal N, offset
            for k in range(K):
                await write_string("Information", wcons[k])
                await write_string("Gaussian", wcons[k])
                await write_integer(yidx, wcons[k])
                await write_vector(beta1, wcons[k])
                
                # Read data from remote sites
                N_remote = await read_vector(rcons[k])
                N += N_remote[0]
                offset_remote = await read_vector(rcons[k])
                offset += offset_remote
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())
    
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

def AVGMLSNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3,
             rcons: Optional[List] = None, wcons: Optional[List] = None) -> Dict[str, Any]:
    """
    Python implementation of the R AVGMLSNet function
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    idx : List[int]
        Indices of samples to use
    yidx : int
        Index of target variable
    lam : float
        Regularization parameter
    rcons, wcons : communication channels for distributed processing
        
    Returns:
    --------
    Dict with AVGMLS network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    
    XX = X.T @ X + np.diag([n * lam] * p)
    Xy = X.T @ y
    yy = np.sum(y**2)
    cXX = linalg.cholesky(XX, lower=False)
    iXX = linalg.cho_solve((cXX, False), np.eye(p))
    beta = iXX @ Xy
    SSE = yy + np.sum(beta * (XX @ beta - 2 * Xy))
    iFisher = iXX @ XX @ iXX
    
    AVGM = {}
    AVGM['beta'] = beta * n
    AVGM['vcov'] = iFisher * (n**2)
    AVGM['SSE'] = SSE
    AVGM['N'] = n
    
    K = 0 if rcons is None else len(rcons)
    
    if K > 0:
        # Using the communication functions from transfer.py
        import asyncio
        from Core.transfer import write_string, write_integer, read_vector, read_matrix
        
        # Need to run this in an async context
        async def communicate():
            nonlocal AVGM
            for k in range(K):
                await write_string("Information", wcons[k])
                await write_string("Gaussian", wcons[k])
                await write_integer(yidx, wcons[k])
                
                # Read data from remote sites
                beta_remote = await read_vector(rcons[k])
                iFisher_remote = await read_matrix(rcons[k])
                SSE_remote = await read_vector(rcons[k])
                n_remote = await read_vector(rcons[k])
                
                n_val = n_remote[0]
                AVGM['beta'] += beta_remote * n_val
                AVGM['vcov'] += iFisher_remote * (n_val**2)
                AVGM['SSE'] += SSE_remote[0]
                AVGM['N'] += n_val
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())
    
    AVGM['beta'] = AVGM['beta'] / AVGM['N']
    AVGM['vcov'] = AVGM['vcov'] / (AVGM['N']**2)
    
    return AVGM
