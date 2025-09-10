Logit = function(X,y,offset=rep(0,ncol(X)),beta0=rep(0,ncol(X)),lam=1e-3,maxiter=100)
{
  p = ncol(X)
  n = nrow(X)
  
  beta = beta0
  iter = 0
  while ( iter < maxiter )
  {
    iter = iter + 1
    
    xb = drop(X%*%beta)
    pr = 1/(1+exp(-xb))
    H = t(X)%*%(X*pr*(1-pr)) + diag(n*lam,p)
    g = t(X)%*%(y-pr) + n*offset - n*lam*beta
    Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) + sum(n*offset*beta) - n*lam*sum(beta^2)/2
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
      nQ = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5]) + sum(n*offset*nbeta) - n*lam*sum(nbeta^2)/2
      if ( nQ-Q > m*step/2 )
        break
      step = step / 2
    }
    
    if ( max(abs(nbeta-beta)) < 1e-5 )
      break
    beta = nbeta
  }
  
  list(beta=beta,H=H)
}



SILogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]

  K = length(rcons)
  
  beta = beta0
  iter = 0
  while ( iter < maxiter )
  {
    iter = iter + 1
    
    xb = drop(X%*%beta)
    pr = 1/(1+exp(-xb))
    H = t(X)%*%(X*pr*(1-pr))
    g = t(X)%*%(y-pr)
    Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
    
    N = n
    if ( K > 0 )
      for ( k in 1:K )
      {
        writeBin("Information",wcons[[k]])
        writeBin("logistic",wcons[[k]])
        writeBin(as.integer(yidx),wcons[[k]])
        writeBin(as.integer(1),wcons[[k]])
        writeVec(as.numeric(beta),wcons[[k]])
        N = N + readVec(rcons[[k]])
        H = H + readMat(rcons[[k]])
        g = g + readVec(rcons[[k]])
        Q = Q + readVec(rcons[[k]])
      }
    
    H = H + diag(N*lam,p)
    g = g - N*lam*beta
    Q = Q - N*lam*sum(beta^2)/2
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
      nQ = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
      if ( K > 0 )
        for ( k in 1:K )
        {
          writeBin("Information",wcons[[k]])
          writeBin("logistic",wcons[[k]])
          writeBin(as.integer(yidx),wcons[[k]])
          writeBin(as.integer(2),wcons[[k]])
          writeVec(as.numeric(nbeta),wcons[[k]])
          readVec(rcons[[k]])
          nQ = nQ + readVec(rcons[[k]])
        }
      nQ = nQ - N*lam*sum(nbeta^2)/2

      if ( nQ-Q > m*step/2 )
        break
      step = step / 2
    }
    
    if ( max(abs(nbeta-beta)) < 1e-5 )
      break
    beta = nbeta
  }
  
  list(beta=beta,H=H)
}



CSLLogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]
  
  K = length(rcons)
  
  
  fit1 = Logit(X,y,beta0=beta0)
  beta1 = fit1$beta

  N = n
  offset = rep(0,p)
  K = length(rcons)
  if ( K > 0 )
    for ( k in 1:K )
    {
      writeBin("Information",wcons[[k]])
      writeBin("logistic",wcons[[k]])
      writeBin(as.integer(yidx),wcons[[k]])
      writeVec(beta1,wcons[[k]])
      N = N + readVec(rcons[[k]])
      offset = offset + readVec(rcons[[k]])
    }
  
  
  fit = Logit(X,y,offset,beta0=beta1)
  fit
}


AVGMLogitNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL,beta0=rep(0,ncol(D)-1),maxiter=100)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]
  
  fit = Logit(X,y,lam=lam,beta0=beta0)
  
  AVGM = list()
  AVGM$beta = fit$beta*n
  AVGM$vcov = chol2inv(chol(fit$H))*n^2
  AVGM$n = n
  
  K = length(hosts)
  for ( k in 1:K )
  {
    writeBin("Information",wcons[[k]])
    writeBin("logistic",wcons[[k]])
    writeBin(as.integer(yidx),wcons[[k]])
    
    beta = readVec(rcons[[k]])
    H = readMat(rcons[[k]])
    n = readVec(rcons[[k]])
    
    AVGM$beta = AVGM$beta + beta*n
    AVGM$vcov = AVGM$vcov + chol2inv(chol(H))*n^2
    AVGM$n = AVGM$n + n
  }
  AVGM$beta = AVGM$beta/AVGM$n
  AVGM$vcov = AVGM$vcov/AVGM$n^2
  
  AVGM
}


ImputeLogit = function(yidx,alpha,wcons)
{
  K = length(wcons)
  for ( k in 1:K )
  {
    writeBin("Impute",wcons[[k]])
    writeBin("logistic",wcons[[k]])
    writeBin(as.integer(yidx),wcons[[k]])
    writeBin(as.numeric(alpha),wcons[[k]])
  }
}

