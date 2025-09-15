# Function CSLMICentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")



CSLMICentral = function(D,M,mvar,method,hosts,ports,cent_ports)
{
  n = nrow(D)
  p = ncol(D)-1
  miss = is.na(D[,mvar])
  nm = sum(miss)
  nc = n-nm
  
  X = D[!miss,-mvar]
  y = D[!miss,mvar]
  
  if ( method == "Gaussian" )
    CSL = CSLCentralLS(X,y,hosts,ports,cent_ports)
  else if ( method == "logistic" )
    CSL = CSLCentralLogit(X,y,hosts,ports,cent_ports)

  imp = NULL
  for ( m in 1:M )
  {
    if ( method == "Gaussian" )
    {
      sig = sqrt(1/rgamma(1,(CSL$N+1)/2,(CSL$SSE+1)/2))
      alpha = CSL$beta + sig * backsolve(CSL$cgram,rnorm(p))
      D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
    }
    else if ( method == "logistic" )
    {
      alpha = CSL$beta + backsolve(CSL$cfisher,rnorm(p))
      pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
      D[miss,mvar] = rbinom(nm,1,pr)
    }          
    
    imp[[m]] = D
  }
  
  imp
}



CSLCentralLS = function(X,y,hosts,ports,cent_ports,lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)
  K = length(hosts)
  
  XX = t(X)%*%X + diag(n*lam,p)
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  cXX = chol(XX)
  iXX = chol2inv(cXX)
  beta1 = drop(iXX%*%Xy)
  
  N = n
  offset = rep(0,p)
  for ( k in 1:K )
  {
    rcon <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("Gaussian",rcon)
    writeBin(beta1,rcon)
    
    wcon <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")

    nk = readBin(wcon,integer())
    gk = readBin(wcon,numeric(),p)
    offset = offset + gk
    N = N + nk
    
    close(rcon)
    close(wcon)
  }
  
  CSL = list()
  CSL$beta = drop(iXX%*%(Xy-n*offset))
  CSL$N = N
  e = drop(y - X%*%CSL$beta)
  CSL$SSE = sum(e^2)*N/n
  gram = XX*N/n
  CSL$cgram = chol(gram)
  
  CSL
}

CSLCentralLogit = function(X,y,hosts,ports,cent_ports,lam=1e-3,maxiter=100)
{
  p = ncol(X)
  n = nrow(X)
  K = length(hosts)

  p = ncol(X)
  n = nrow(X)
  K = length(hosts)
  
  fit = Logit(X,y,lam=lam,maxiter=maxiter)
  
  N = n
  offset = rep(0,p)
  for ( k in 1:K )
  {
    rcon <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("logistic",rcon)
    writeBin(fit$beta,rcon)
    
    wcon <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
    nk = readBin(wcon,integer())
    gk = readBin(wcon,numeric(),p)
    
    offset = offset + gk
    N = N + nk
    
    close(rcon)
    close(wcon)
  }
  
  fit = Logit(X,y,offset,lam=lam,maxiter=maxiter)
  
  CSL = list()
  CSL$beta = fit$beta
  CSL$cfisher = chol(fit$H*N/n)
  
  CSL  
}
