"""
Python implementation of SIMICentral.R

Original R function description:
SIMICentral = function(D,M,mvar,method,hosts,ports,cent_ports)

Arguments:
D: Data matrix
M: Number of imputations
mvar: Index of missing variable
method: "Gaussian" or "logistic" depending on missing data type
hosts: a vector of hostnames of remote sites
ports: a vector of ports of remote sites
cent_ports: a vector of local listening ports dedicated to corresponding remote sites
"""

import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict, Tuple, Optional
import scipy.stats as stats
from scipy.linalg import cholesky, cho_solve
from scipy.special import expit
from Core.transfer import (
    read_matrix, write_matrix, read_vector, write_vector, 
    write_string, read_string, write_integer, read_integer,
    WebSocketWrapper, get_wrapped_websocket
)

# Function to set expected remote sites
# Dictionary to store WebSocket connections for remote sites
remote_websockets: Dict[str, WebSocket] = {}

# Flag to indicate when imputation is running (to avoid concurrent WebSocket access)
imputation_running = asyncio.Event()

# Dictionary to store locks for each site to ensure sequential communication
site_locks = {}

async def simi_central(D: np.ndarray, config: dict = None, site_ids: List[str] = None, websockets: Dict[str, WebSocket] = None):
    """
    Python implementation of SIMICentral with unified interface
    
    Parameters:
    -----------
    D : np.ndarray
        Data matrix
    config : dict
        Configuration parameters including:
        - M: Number of imputations
        - mvar: Index of missing variable (0-based)
        - method: "Gaussian" or "logistic"
        - Other optional parameters
    site_ids : List[str]
        List of remote site IDs
    websockets : Dict[str, WebSocket]
        Dictionary of WebSocket connections to remote sites
    
    R equivalent:
    SIMICentral = function(D,M,mvar,method,hosts,ports,cent_ports)
    {
      n = nrow(D)
      p = ncol(D)-1
      miss = is.na(D[,mvar])
      nm = sum(miss)
      nc = n-nm
      
      X = D[!miss,-mvar]
      y = D[!miss,mvar]
      
      if ( method == "Gaussian" )
      {
        SI = SICentralLS(X,y,hosts,ports,cent_ports)
        cvcov = chol(SI$vcov)
      }
      else if ( method == "logistic" )
      {
        SI = SICentralLogit(X,y,hosts,ports,cent_ports)
        cvcov = chol(SI$vcov)
      }
      
      imp = NULL
      for ( m in 1:M )
      {
        if ( method == "Gaussian" )
        {
          sig = sqrt(1/rgamma(1,(SI$n+1)/2,(SI$SSE+1)/2))
          alpha = SI$beta + sig * t(cvcov)%*%rnorm(p)
          D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
        }
        else if ( method == "logistic" )
        {
          alpha = SI$beta + t(cvcov)%*%rnorm(p)
          pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
          D[miss,mvar] = rbinom(nm,1,pr)
        }          
        
        imp[[m]] = D
      }
      
      imp
    }
    """
    # Extract parameters from config
    if config is None:
        raise ValueError("Config dictionary is required")
    
    M = config.get("M")
    mvar = config.get("mvar")
    method = config.get("method")
    
    if M is None or mvar is None or method is None:
        raise ValueError("Missing required parameters in config: M, mvar, or method")
    # Use provided websockets if available
    global remote_websockets
    if websockets is not None:
        remote_websockets = websockets
        print(f"Using provided WebSocket connections for {len(remote_websockets)} sites")
    
    # Create locks for each site if they don't exist
    for site_id in site_ids:
        if site_id not in site_locks:
            site_locks[site_id] = asyncio.Lock()
    
    # Set the imputation_running flag before starting imputation
    imputation_running.set()
    print("Imputation started, flag set")
    
    try:
        # Use the existing connections, no need to wait for new connections
        print("Using existing WebSocket connections, beginning communication")
        
        n, p_plus_1 = D.shape
        p = p_plus_1 - 1
        miss = np.isnan(D[:, mvar])
        nm = np.sum(miss)
        nc = n - nm
        
        X = np.delete(D[~miss], mvar, axis=1)
        y = D[~miss, mvar]
        
        if method == "Gaussian":
            SI = await si_central_ls(X, y, site_ids, websockets=websockets)
            # Safely compute Cholesky decomposition of vcov
            try:
                # Check for NaN/Inf values
                if np.isnan(SI["vcov"]).any() or np.isinf(SI["vcov"]).any():
                    print("Warning: NaN or Inf values in vcov, applying regularization")
                    SI["vcov"] = np.nan_to_num(SI["vcov"])
                    # Add regularization to diagonal
                    reg = max(1e-8 * np.trace(SI["vcov"]) / SI["vcov"].shape[0], 1e-6)
                    for i in range(SI["vcov"].shape[0]):
                        SI["vcov"][i, i] += reg
                
                cvcov = cholesky(SI["vcov"], lower=True)
            except np.linalg.LinAlgError:
                print("Cholesky decomposition of vcov failed, using stronger regularization")
                reg = max(1e-4 * np.trace(SI["vcov"]) / SI["vcov"].shape[0], 1e-2)
                vcov_reg = SI["vcov"] + np.eye(SI["vcov"].shape[0]) * reg
                cvcov = cholesky(vcov_reg, lower=True)
        elif method == "logistic":
            SI = await si_central_logit(X, y, site_ids, websockets=websockets)
            # Safely compute Cholesky decomposition of vcov
            try:
                # Check for NaN/Inf values
                if np.isnan(SI["vcov"]).any() or np.isinf(SI["vcov"]).any():
                    print("Warning: NaN or Inf values in vcov, applying regularization")
                    SI["vcov"] = np.nan_to_num(SI["vcov"])
                    # Add regularization to diagonal
                    reg = max(1e-8 * np.trace(SI["vcov"]) / SI["vcov"].shape[0], 1e-6)
                    for i in range(SI["vcov"].shape[0]):
                        SI["vcov"][i, i] += reg
                
                cvcov = cholesky(SI["vcov"], lower=True)
            except np.linalg.LinAlgError:
                print("Cholesky decomposition of vcov failed, using stronger regularization")
                reg = max(1e-4 * np.trace(SI["vcov"]) / SI["vcov"].shape[0], 1e-2)
                vcov_reg = SI["vcov"] + np.eye(SI["vcov"].shape[0]) * reg
                cvcov = cholesky(vcov_reg, lower=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        imp = []
        for m in range(M):
            D_imputed = D.copy()
            
            if method == "Gaussian":
                # sqrt(1/rgamma(1,(SI$n+1)/2,(SI$SSE+1)/2))
                sig = np.sqrt(1/np.random.gamma((SI["n"]+1)/2, 2/(SI["SSE"]+1)))
                
                # alpha = SI$beta + sig * t(cvcov)%*%rnorm(p)
                alpha = SI["beta"] + sig * np.dot(cvcov, np.random.normal(size=p))
                
                # D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
                D_imputed[miss, mvar] = np.dot(np.delete(D[miss], mvar, axis=1), alpha) + np.random.normal(0, sig, nm)
            
            elif method == "logistic":
                # alpha = SI$beta + t(cvcov)%*%rnorm(p)
                alpha = SI["beta"] + np.dot(cvcov, np.random.normal(size=p))
                
                # pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
                pr = expit(np.dot(np.delete(D[miss], mvar, axis=1), alpha))
                
                # D[miss,mvar] = rbinom(nm,1,pr)
                D_imputed[miss, mvar] = np.random.binomial(1, pr)
            
            imp.append(D_imputed)
        
        return imp
    finally:
        # Clear the imputation_running flag when done
        imputation_running.clear()
        print("Imputation completed, flag cleared")

async def si_central_ls(X: np.ndarray, y: np.ndarray, site_ids: List[str], lam: float = 1e-3, websockets: Dict[str, WebSocket] = None):
    """
    Python implementation of SICentralLS using JSON-only WebSocket communication
    
    R equivalent:
    SICentralLS = function(X,y,hosts,ports,cent_ports,lam=1e-3)
    {
      p = ncol(X)
      n = nrow(X)
      
      XX = t(X)%*%X
      Xy = drop(t(X)%*%y)
      yy = sum(y^2)
      
      K = length(hosts)
      for ( k in 1:K )
      {
        rcon <- socketConnection(hosts[k],ports[k],open="w+b")
        writeBin("Gaussian",rcon)
        wcon <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
        
        n = n + readVec(wcon)
        XX = XX + readMat(wcon)
        Xy = Xy + readVec(wcon)
        yy = yy + readVec(wcon)
        
        close(rcon)
        close(wcon)
      }
      
      SI = list()
      cXX = chol(XX)
      iXX = chol2inv(cXX)
      SI$beta = drop(iXX%*%Xy)
      SI$vcov = iXX
      SI$SSE = yy + sum(SI$beta*(XX%*%SI$beta-2*Xy))
      SI$n = n
      
      SI
    }
    """
    p = X.shape[1]
    n = X.shape[0]
    
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    yy = np.sum(y**2)
    
    # Use provided websockets if available
    global remote_websockets
    ws_dict = websockets if websockets is not None else remote_websockets
    
    # Send method to all remote sites and gather their contributions
    # Process one site at a time for a more synchronous pattern
    for site_id in site_ids:
        # Acquire lock for this site to ensure sequential access
        async with site_locks[site_id]:
            ws = ws_dict[site_id]
            
            # Send method as a string
            await write_string("Gaussian", ws)
            
            # Receive sample size
            remote_n = await read_vector(ws)
            n += int(remote_n[0])
            
            # Receive matrix XX
            remote_XX = await read_matrix(ws)
            XX += remote_XX
            
            # Receive vector Xy
            remote_Xy = await read_vector(ws)
            Xy += remote_Xy
            
            # Receive scalar yy
            remote_yy = await read_vector(ws)
            yy += remote_yy[0]
            
            print(f"Received data from site {site_id}: n={int(remote_n[0])}, XX shape={remote_XX.shape}")
    
    # Calculate results
    SI = {}
    
    # Add regularization to ensure positive definiteness and check for NaN/Inf values
    print(f"Matrix XX shape: {XX.shape}")
    print(f"Contains NaN: {np.isnan(XX).any()}, Contains Inf: {np.isinf(XX).any()}")
    
    # Replace any NaN or Inf values with zeros
    XX = np.nan_to_num(XX)
    Xy = np.nan_to_num(Xy)
    
    # Add a small regularization term to the diagonal to ensure positive definiteness
    reg_param = max(1e-10 * np.trace(XX) / XX.shape[0], 1e-6)
    print(f"Adding regularization parameter: {reg_param}")
    
    for i in range(XX.shape[0]):
        XX[i, i] += reg_param
    
    try:
        # Attempt Cholesky decomposition
        cXX = cholesky(XX, lower=True)
    except np.linalg.LinAlgError:
        # If Cholesky fails, try a stronger regularization
        print("Cholesky decomposition failed, increasing regularization")
        reg_param = max(1e-4 * np.trace(XX) / XX.shape[0], 1e-2)
        for i in range(XX.shape[0]):
            XX[i, i] += reg_param
        print(f"New regularization parameter: {reg_param}")
        # Try again with stronger regularization
        cXX = cholesky(XX + np.eye(XX.shape[0]) * reg_param, lower=True)
    
    # Inverse of XX using the Cholesky decomposition with error handling
    try:
        iXX = cho_solve((cXX, True), np.eye(p))
    except Exception as e:
        print(f"Error in cho_solve: {str(e)}")
        # Fallback to pseudoinverse
        print("Using fallback pseudoinverse")
        iXX = np.linalg.pinv(XX)
    
    # Check for NaN/Inf values in iXX
    if np.isnan(iXX).any() or np.isinf(iXX).any():
        print("Warning: NaN or Inf values in iXX after inversion, using pseudoinverse")
        iXX = np.linalg.pinv(XX)
    
    # Beta calculation - with NaN/Inf checks
    Xy = np.nan_to_num(Xy)  # Ensure Xy is clean
    beta = np.dot(iXX, Xy)
    
    # Check for NaN/Inf in beta
    if np.isnan(beta).any() or np.isinf(beta).any():
        print("Warning: NaN or Inf values in beta, replacing with zeros")
        beta = np.nan_to_num(beta)
    
    SI["beta"] = beta
    SI["vcov"] = iXX
    
    # Calculate SSE safely
    try:
        SSE = yy + np.sum(beta * (np.dot(XX, beta) - 2 * Xy))
        if np.isnan(SSE) or np.isinf(SSE):
            print("Warning: SSE is NaN or Inf, using fallback")
            SSE = np.abs(yy)  # Fallback value
    except Exception as e:
        print(f"Error calculating SSE: {str(e)}")
        SSE = np.abs(yy)  # Fallback value
        
    SI["SSE"] = SSE
    SI["n"] = n
    
    return SI

async def si_central_logit(X: np.ndarray, y: np.ndarray, site_ids: List[str], lam: float = 1e-3, maxiter: int = 100, websockets: Dict[str, WebSocket] = None):
    """
    Python implementation of SICentralLogit using JSON-only WebSocket communication
    
    R equivalent:
    SICentralLogit = function(X,y,hosts,ports,cent_ports,lam=1e-3,maxiter=100)
    {
      p = ncol(X)
      n = nrow(X)
      
      K = length(hosts)
      rcons = list()
      wcons = list()
      N = n
      for ( k in 1:K )
      {
        rcons[[k]] <- socketConnection(hosts[k],ports[k],open="w+b")
        writeBin("logistic",rcons[[k]])
        wcons[[k]] <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
        N = N + readVec(wcons[[k]])
      }
      
      beta = rep(0,p)
      iter = 0
      while ( iter < maxiter )
      {
        iter = iter + 1
        
        xb = drop(X%*%beta)
        pr = 1/(1+exp(-xb))
        H = t(X)%*%(X*pr*(1-pr)) + diag(N*lam,p)
        g = t(X)%*%(y-pr) - N*lam*beta
        Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) - N*lam*sum(beta^2)/2
        for ( k in 1:K )
        {
          writeBin(as.integer(1),rcons[[k]])
          writeVec(beta,rcons[[k]])
          H = H + readMat(wcons[[k]])
          g = g + readVec(wcons[[k]])
          Q = Q + readVec(wcons[[k]])
        }
        dir = drop(chol2inv(chol(H))%*%g)
        m = sum(dir*g)
        
        step = 1
        while (TRUE)
        {
          nbeta = beta + step*dir
          if ( max(abs(nbeta-beta)) < 1e-5 )
            break
          xb = drop(X%*%nbeta)
          pr = 1/(1+exp(-xb))
          nQ = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) - N*lam*sum(nbeta^2)/2
          for ( k in 1:K )
          {
            writeBin(as.integer(2),rcons[[k]])
            writeVec(nbeta,rcons[[k]])
            nQ = nQ + readVec(wcons[[k]])
          }
          if ( nQ-Q > m*step/2 )
            break
          step = step / 2
        }
        
        if ( max(abs(nbeta-beta)) < 1e-5 )
          break
        beta = nbeta
      }
      
      for ( k in 1:K )
      {
        writeBin(as.integer(0),rcons[[k]])
        close(rcons[[k]])
        close(wcons[[k]])
      }
      
      SI = list()
      SI$beta = beta
      SI$vcov = chol2inv(chol(H))

      SI
    }
    """
    p = X.shape[1]
    n = X.shape[0]
    
    # Send method to all remote sites and gather sample sizes - one site at a time
    N = n
    for site_id in site_ids:
        # Acquire lock for this site to ensure sequential access
        async with site_locks[site_id]:
            ws = remote_websockets[site_id]
            
            # Send method using JSON-only protocol
            await write_string("logistic", ws)
            print(f"Sent 'logistic' method to site {site_id}")
            
            # Receive sample size
            remote_n = await read_vector(ws)
            remote_n_value = int(remote_n[0])
            N += remote_n_value
            print(f"Received sample size from site {site_id}: n={remote_n_value}")
    
    # Initialize beta to zeros
    beta = np.zeros(p)
    iter_count = 0
    
    # Main optimization loop
    while iter_count < maxiter:
        iter_count += 1
        print(f"Iteration {iter_count}/{maxiter}")
        
        # Calculate probabilities and statistics for central data
        xb = np.dot(X, beta)
        pr = expit(xb)  # 1/(1+exp(-xb))
        
        # Calculate Hessian with regularization
        weights = pr * (1 - pr)
        H = np.dot(X.T * weights, X) + np.eye(p) * (N * lam)
        
        # Calculate gradient with regularization
        g = np.dot(X.T, (y - pr)) - N * lam * beta
        
        # Calculate objective function Q
        low_pr_mask = pr < 0.5
        high_pr_mask = ~low_pr_mask
        
        Q = np.sum(y * xb)
        if np.any(low_pr_mask):
            Q += np.sum(np.log(np.maximum(1e-10, 1 - pr[low_pr_mask])))
        if np.any(high_pr_mask):
            Q += np.sum(np.log(np.maximum(1e-10, pr[high_pr_mask])) - xb[high_pr_mask])
        Q -= N * lam * np.sum(beta**2) / 2
        
        # Gather contributions from remote sites
        for site_id in site_ids:
            async with site_locks[site_id]:
                ws = remote_websockets[site_id]
                
                # Send mode 1 (calculate H and g)
                await write_integer(1, ws)
                print(f"Sent mode 1 to site {site_id}")
                
                # Send current beta
                await write_vector(beta, ws)
                print(f"Sent beta vector to site {site_id}")
                
                # Receive Hessian matrix
                remote_H = await read_matrix(ws)
                H += remote_H
                print(f"Received Hessian matrix from site {site_id}")
                
                # Receive gradient vector
                remote_g = await read_vector(ws)
                g += remote_g
                print(f"Received gradient vector from site {site_id}")
                
                # Receive Q contribution
                remote_Q = await read_vector(ws)
                Q += remote_Q[0]
                print(f"Received Q value from site {site_id}: {remote_Q[0]}")
        
        # Compute the update direction
        try:
            # Try Cholesky decomposition first for better numerical stability
            cH = cholesky(H, lower=True)  # Use scipy.linalg.cholesky that we imported
            direction = cho_solve((cH, True), g)  # Use scipy.linalg.cho_solve that we imported
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse if Cholesky fails
            print("Warning: Cholesky decomposition failed, using pseudoinverse")
            direction = np.dot(np.linalg.pinv(H), g)
        
        # Compute dot product for line search
        m = np.sum(direction * g)
        
        # Line search
        step = 1.0
        while True:
            # Calculate new beta
            nbeta = beta + step * direction
            
            # Check for convergence
            if np.max(np.abs(nbeta - beta)) < 1e-5:
                break
            
            # Calculate new probabilities and objective function
            xb = np.dot(X, nbeta)
            pr = expit(xb)
            
            low_pr_mask = pr < 0.5
            high_pr_mask = ~low_pr_mask
            
            nQ = np.sum(y * xb)
            if np.any(low_pr_mask):
                nQ += np.sum(np.log(np.maximum(1e-10, 1 - pr[low_pr_mask])))
            if np.any(high_pr_mask):
                nQ += np.sum(np.log(np.maximum(1e-10, pr[high_pr_mask])) - xb[high_pr_mask])
            nQ -= N * lam * np.sum(nbeta**2) / 2
            
            # Gather Q contributions from remote sites
            for site_id in site_ids:
                async with site_locks[site_id]:
                    ws = remote_websockets[site_id]
                    
                    # Send mode 2 (calculate Q only)
                    await write_integer(2, ws)
                    
                    # Send new beta
                    await write_vector(nbeta, ws)
                    
                    # Receive Q contribution
                    remote_nQ = await read_vector(ws)
                    nQ += remote_nQ[0]
            
            # Check Armijo condition
            if nQ - Q > m * step / 2:
                break
            
            # Reduce step size
            step = step / 2
            print(f"Reducing step size to {step}")
            
            # Avoid too small steps
            if step < 1e-10:
                print("Warning: Step size too small, stopping line search")
                break
        
        # Check for convergence
        if np.max(np.abs(nbeta - beta)) < 1e-5:
            print(f"Converged after {iter_count} iterations")
            beta = nbeta
            break
        
        # Update beta
        beta = nbeta
    
    # Send termination signal to all remote sites
    for site_id in site_ids:
        async with site_locks[site_id]:
            ws = remote_websockets[site_id]
            await write_integer(0, ws)
            print(f"Sent termination signal to site {site_id}")
    
    # Calculate final results
    SI = {}
    SI["beta"] = beta
    
    # Safely compute the variance-covariance matrix
    try:
        cH = cholesky(H, lower=True)  # Use scipy.linalg.cholesky
        SI["vcov"] = cho_solve((cH, True), np.eye(p))  # Use scipy.linalg.cho_solve
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Error computing variance-covariance matrix: {e}")
        # Use pseudoinverse as a fallback
        SI["vcov"] = np.linalg.pinv(H)
        
    return SI
    
# This function is now deprecated - we use the central WebSocket handling in run_imputation.py
# The FastAPI app has been removed since it's now managed externally
def run_central_server(host="0.0.0.0", port=8000):
    print("Warning: This function is deprecated. The server should be run through run_imputation.py")
    raise NotImplementedError("Direct server running is no longer supported. Use run_imputation.py instead.")

if __name__ == "__main__":
    print("This module should not be run directly. Use run_imputation.py instead.")
