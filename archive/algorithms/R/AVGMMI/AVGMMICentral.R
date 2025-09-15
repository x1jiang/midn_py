# Function AVGMMICentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")
source("Core/Logit.R")



AVGMMICentral = function(D,M,mvar,method,hosts,ports,cent_ports)
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
    AVGM = AVGMCentralLS(X,y,hosts,ports,cent_ports)
    cvcov = chol(AVGM$vcov)
  }
  else if ( method == "logistic" )
  {
    AVGM = AVGMCentralLogit(X,y,hosts,ports,cent_ports)
    cvcov = chol(AVGM$vcov)
  }
  
  imp = NULL
  for ( m in 1:M )
  {
    if ( method == "Gaussian" )
    {
      sig = sqrt(1/rgamma(1,(AVGM$n+1)/2,(AVGM$SSE+1)/2))
      alpha = AVGM$beta + sig * t(cvcov)%*%rnorm(p)
      D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
    }
    else if ( method == "logistic" )
    {
      alpha = AVGM$beta + t(cvcov)%*%rnorm(p)
      pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
      D[miss,mvar] = rbinom(nm,1,pr)
    }          
    
    imp[[m]] = D
  }
  
  imp
}

AVGMCentralLS = function(X,y,hosts,ports,cent_ports,lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)
  XX = t(X)%*%X + diag(lam,p)
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  cXX = chol(XX)
  iXX = chol2inv(cXX)
  beta = drop(iXX%*%Xy)
  SSE = yy + sum(beta*(XX%*%beta-2*Xy))
  iFisher = iXX%*%XX%*%iXX
  
  AVGM = list()
  AVGM$beta = beta*n
  AVGM$vcov = iFisher*n^2
  AVGM$SSE = SSE
  AVGM$n = n
  
  K = length(hosts)
  for ( k in 1:K )
  {
    rcon <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("Gaussian",rcon)
    wcon <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
    
    beta = readVec(wcon)
    iFisher = readMat(wcon)
    SSE = readVec(wcon)
    n = readVec(wcon)
    
    close(rcon)
    close(wcon)
    
    AVGM$beta = AVGM$beta + beta*n
    AVGM$vcov = AVGM$vcov + iFisher*n^2
    AVGM$SSE = AVGM$SSE + SSE
    AVGM$n = AVGM$n + n
  }
  AVGM$beta = AVGM$beta/AVGM$n
  AVGM$vcov = AVGM$vcov/AVGM$n^2
  
  AVGM
}

AVGMCentralLogit = function(X,y,hosts,ports,cent_ports,lam=1e-3,maxiter=100)
{
  p = ncol(X)
  n = nrow(X)
  
  fit = Logit(X,y,lam=lam,maxiter=maxiter)
  
  AVGM = list()
  AVGM$beta = fit$beta*n
  AVGM$vcov = chol2inv(chol(fit$H))*n^2
  AVGM$n = n
  
  K = length(hosts)
  for ( k in 1:K )
  {
    rcon <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("logistic",rcon)
    wcon <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
    
    beta = readVec(wcon)
    H = readMat(wcon)
    n = readVec(wcon)
    
    close(rcon)
    close(wcon)
    
    AVGM$beta = AVGM$beta + beta*n
    AVGM$vcov = AVGM$vcov + chol2inv(chol(H))*n^2
    AVGM$n = AVGM$n + n
  }
  AVGM$beta = AVGM$beta/AVGM$n
  AVGM$vcov = AVGM$vcov/AVGM$n^2
  
  AVGM
}

