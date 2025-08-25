# Function IMICentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type



IMICentral = function(D,M,mvar,method)
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
    I = IMICentralLS(X,y)
    cvcov = chol(I$vcov)
  }
  else if ( method == "logistic" )
  {
    I = IMICentralLogit(X,y)
    cvcov = chol(I$vcov)
  }
  
  imp = NULL
  for ( m in 1:M )
  {
    if ( method == "Gaussian" )
    {
      sig = sqrt(1/rgamma(1,(I$n+1)/2,(I$SSE+1)/2))
      alpha = I$beta + sig * t(cvcov)%*%rnorm(p)
      D[miss,mvar] = D[miss,-mvar] %*% alpha + rnorm(nm,0,sig)
    }
    else if ( method == "logistic" )
    {
      alpha = I$beta + t(cvcov)%*%rnorm(p)
      pr = 1 / (1 + exp(-D[miss,-mvar] %*% alpha))
      D[miss,mvar] = rbinom(nm,1,pr)
    }          
    
    imp[[m]] = D
  }
  
  imp
}



IMICentralLS = function(X,y,lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)
  
  XX = t(X)%*%X
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  
  I = list()
  cXX = chol(XX)
  iXX = chol2inv(cXX)
  I$beta = drop(iXX%*%Xy)
  I$vcov = iXX
  I$SSE = yy + sum(I$beta*(XX%*%I$beta-2*Xy))
  I$n = n
  
  I
}

IMICentralLogit = function(X,y,lam=1e-3,maxiter=100)
{
  p = ncol(X)
  n = nrow(X)
  
  rcons = list()
  wcons = list()
  N = n

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
      if ( nQ-Q > m*step/2 )
        break
      step = step / 2
    }
    
    if ( max(abs(nbeta-beta)) < 1e-5 )
      break
    beta = nbeta
  }
  
  I = list()
  I$beta = beta
  I$vcov = chol2inv(chol(H))
  
  I
}
