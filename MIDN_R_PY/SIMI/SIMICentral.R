# Function SIMICentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")



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
