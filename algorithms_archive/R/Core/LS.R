LS = function(X,y,offset=rep(0,ncol(X)),lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)

  XX = t(X)%*%X + diag(lam,p)
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  
  cXX = chol(XX)
  iXX = chol2inv(cXX)
  
  LS = list()
  LS$beta = drop(iXX%*%(Xy-n*offset))
  LS$n = n
  e = drop(y - X%*%LS$beta)
  LS$SSE = sum(e^2)
  LS$cgram = cXX
  
  LS
}

SILSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]
  XX = t(X)%*%X
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  
  N = n
  K = length(rcons)
  if ( K > 0 )
    for ( k in 1:K )
    {
      writeBin("Information",wcons[[k]])
      writeBin("Gaussian",wcons[[k]])
      writeBin(as.integer(yidx),wcons[[k]])
      N = N + readVec(rcons[[k]])
      XX = XX + readMat(rcons[[k]])
      Xy = Xy + readVec(rcons[[k]])
      yy = yy + readVec(rcons[[k]])
    }
  
  cXX = chol(XX+diag(N*lam,p))
  iXX = chol2inv(cXX)
  
  SILS = list()
  SILS$beta = drop(iXX%*%Xy)
  SILS$N = N
  SILS$SSE = yy-crossprod(SILS$beta,2*Xy-XX%*%SILS$beta)
  SILS$cgram = cXX
  
  SILS
}


ImputeLS = function(yidx,beta,sig,wcons)
{
  K = length(wcons)
  for ( k in 1:K )
  {
    writeBin("Impute",wcons[[k]])
    writeBin("Gaussian",wcons[[k]])
    writeBin(as.integer(yidx),wcons[[k]])
    writeBin(as.numeric(beta),wcons[[k]])
    writeBin(as.numeric(sig),wcons[[k]])
  }
}


CSLLSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]

  XX = t(X)%*%X + diag(n*lam,p)
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  cXX = chol(XX)
  iXX = chol2inv(cXX)
  beta1 = drop(iXX%*%Xy)
  
  N = n
  offset = rep(0,p)
  K = length(rcons)
  if ( K > 0 )
    for ( k in 1:K )
    {
      writeBin("Information",wcons[[k]])
      writeBin("Gaussian",wcons[[k]])
      writeBin(as.integer(yidx),wcons[[k]])
      writeVec(beta1,wcons[[k]])
      N = N + readVec(rcons[[k]])
      offset = offset + readVec(rcons[[k]])
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


AVGMLSNet = function(D,idx,yidx,lam=1e-3,rcons=NULL,wcons=NULL)
{
  p = ncol(D)-1
  n = length(idx)
  
  X = matrix(D[idx,-yidx],n,p)
  y = D[idx,yidx]
  
  XX = t(X)%*%X + diag(n*lam,p)
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
  AVGM$N = n
  
  K = length(rcons)
  for ( k in 1:K )
  {
    writeBin("Information",wcons[[k]])
    writeBin("Gaussian",wcons[[k]])
    writeBin(as.integer(yidx),wcons[[k]])
    
    beta = readVec(rcons[[k]])
    iFisher = readMat(rcons[[k]])
    SSE = readVec(rcons[[k]])
    n = readVec(rcons[[k]])

    AVGM$beta = AVGM$beta + beta*n
    AVGM$vcov = AVGM$vcov + iFisher*n^2
    AVGM$SSE = AVGM$SSE + SSE
    AVGM$N = AVGM$N + n
  }
  AVGM$beta = AVGM$beta/AVGM$N
  AVGM$vcov = AVGM$vcov/AVGM$N^2
  
  AVGM
}

