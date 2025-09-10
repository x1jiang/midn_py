import numpy as np
from scipy import linalg
from typing import List, Dict, Any, Optional, Union, Tuple

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

def SILogitNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3, 
              rcons: Optional[List] = None, wcons: Optional[List] = None, 
              beta0: Optional[np.ndarray] = None, maxiter: int = 100) -> Dict[str, Any]:
    """
    Python implementation of the R SILogitNet function
    
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
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    maxiter : int
        Maximum number of iterations
        
    Returns:
    --------
    Dict with SILogit network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    
    if beta0 is None:
        beta0 = np.zeros(p)
        
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    
    K = 0 if rcons is None else len(rcons)
    
    beta = beta0.copy()
    iter_count = 0
    while iter_count < maxiter:
        iter_count += 1
        
        xb = X @ beta
        pr = 1 / (1 + np.exp(-xb))
        H = X.T @ (X * pr[:, np.newaxis] * (1 - pr[:, np.newaxis]))
        g = X.T @ (y - pr)
        
        # Calculate Q - log likelihood
        Q = np.sum(y * xb)
        Q += np.sum(np.log(1 - pr[pr < 0.5] + 1e-10))
        Q += np.sum(np.log(pr[pr >= 0.5] + 1e-10) - xb[pr >= 0.5])
        
        N = n
        if K > 0:
            # Using the communication functions from transfer.py
            import asyncio
            from Core.transfer import write_string, write_integer, write_vector, read_vector, read_matrix
            
            # Need to run this in an async context
            async def communicate_info():
                nonlocal N, H, g, Q
                for k in range(K):
                    await write_string("Information", wcons[k])
                    await write_string("logistic", wcons[k])
                    await write_integer(yidx, wcons[k])
                    await write_integer(1, wcons[k])
                    await write_vector(beta, wcons[k])
                    
                    # Read data from remote sites
                    N_remote = await read_vector(rcons[k])
                    N += N_remote[0]
                    H_remote = await read_matrix(rcons[k])
                    H += H_remote
                    g_remote = await read_vector(rcons[k])
                    g += g_remote
                    Q_remote = await read_vector(rcons[k])
                    Q += Q_remote[0]
            
            # Execute the async function in the current event loop
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(communicate_info())
            except RuntimeError:
                # If there's no event loop in the current thread
                asyncio.run(communicate_info())
        
        H = H + np.diag([N * lam] * p)
        g = g - N * lam * beta
        Q = Q - N * lam * np.sum(beta**2) / 2
        
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
            
            if K > 0:
                # Using the communication functions from transfer.py
                import asyncio
                from Core.transfer import write_string, write_integer, write_vector, read_vector
                
                # Need to run this in an async context
                async def communicate_update():
                    nonlocal nQ
                    for k in range(K):
                        await write_string("Information", wcons[k])
                        await write_string("logistic", wcons[k])
                        await write_integer(yidx, wcons[k])
                        await write_integer(2, wcons[k])
                        await write_vector(nbeta, wcons[k])
                        
                        # Read data from remote sites
                        _ = await read_vector(rcons[k])  # Discard this value
                        nQ_remote = await read_vector(rcons[k])
                        nQ += nQ_remote[0]
                
                # Execute the async function in the current event loop
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(communicate_update())
                except RuntimeError:
                    # If there's no event loop in the current thread
                    asyncio.run(communicate_update())
                    
            nQ = nQ - N * lam * np.sum(nbeta**2) / 2
            
            if nQ - Q > m * step / 2:
                break
            step = step / 2
        
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            break
        beta = nbeta
    
    return {'beta': beta, 'H': H}

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

def CSLLogitNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3, 
               rcons: Optional[List] = None, wcons: Optional[List] = None, 
               beta0: Optional[np.ndarray] = None, maxiter: int = 100) -> Dict[str, Any]:
    """
    Python implementation of the R CSLLogitNet function
    
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
    beta0 : np.ndarray, optional
        Initial coefficients, defaults to zeros
    maxiter : int
        Maximum number of iterations
        
    Returns:
    --------
    Dict with CSLLogit network results
    """
    p = D.shape[1] - 1
    n = len(idx)
    
    if beta0 is None:
        beta0 = np.zeros(p)
    
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    
    K = 0 if rcons is None else len(rcons)
    
    fit1 = Logit(X, y, beta0=beta0)
    beta1 = fit1['beta']
    
    N = n
    offset = np.zeros(p)
    
    if K > 0:
        # Using the communication functions from transfer.py
        import asyncio
        from Core.transfer import write_string, write_integer, write_vector, read_vector
        
        # Need to run this in an async context
        async def communicate():
            nonlocal N, offset
            for k in range(K):
                await write_string("Information", wcons[k])
                await write_string("logistic", wcons[k])
                await write_integer(yidx, wcons[k])
                await write_integer(0, wcons[k])       # HIGHLIGHT: mode=0 means "CSL offset request"
                await write_vector(beta1, wcons[k])    # beta1 used to compute pr for offset                
                
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

def AVGMLogitNet(D: np.ndarray, idx: List[int], yidx: int, lam: float = 1e-3, 
                rcons: Optional[List] = None, wcons: Optional[List] = None, 
                beta0: Optional[np.ndarray] = None, maxiter: int = 100) -> Dict[str, Any]:
    """
    Python implementation of the R AVGMLogitNet function
    
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
    
    if beta0 is None:
        beta0 = np.zeros(p)
    
    X = np.delete(D[idx, :], yidx, axis=1)
    y = D[idx, yidx]
    
    fit = Logit(X, y, lam=lam, beta0=beta0)
    
    AVGM = {}
    try:
        AVGM['beta'] = fit['beta'] * n
        AVGM['vcov'] = linalg.cho_solve((linalg.cholesky(fit['H'], lower=False), False), np.eye(p)) * (n**2)
        AVGM['n'] = n
    except np.linalg.LinAlgError:
        # Fall back to a more robust method if Cholesky decomposition fails
        AVGM['beta'] = fit['beta'] * n
        AVGM['vcov'] = np.linalg.inv(fit['H']) * (n**2)
        AVGM['n'] = n
    
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
                await write_string("logistic", wcons[k])
                await write_integer(yidx, wcons[k])
                
                # Read data from remote sites
                beta_remote = await read_vector(rcons[k])
                H_remote = await read_matrix(rcons[k])
                n_remote = await read_vector(rcons[k])
                
                n_val = n_remote[0]
                
                try:
                    vcov_remote = linalg.cho_solve((linalg.cholesky(H_remote, lower=False), False), np.eye(p))
                except np.linalg.LinAlgError:
                    # Fall back to a more robust method if Cholesky decomposition fails
                    vcov_remote = np.linalg.inv(H_remote)
                
                AVGM['beta'] += beta_remote * n_val
                AVGM['vcov'] += vcov_remote * (n_val**2)
                AVGM['n'] += n_val
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())
        
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

def ImputeLogit(yidx: int, alpha: float, wcons: List) -> None:
    """
    Python implementation of the R ImputeLogit function
    
    Parameters:
    -----------
    yidx : int
        Index of target variable
    alpha : float
        Alpha value
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
                await write_string("logistic", wcons[k])
                await write_integer(yidx, wcons[k])
                await write_vector(alpha, wcons[k])  # HIGHLIGHT: send the full coefficient vector
                
        
        # Execute the async function in the current event loop
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(communicate())
        except RuntimeError:
            # If there's no event loop in the current thread
            asyncio.run(communicate())
