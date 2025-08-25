# Function AVGMMIRemote
# Arguments:
# D: Data matrix
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site


source("Core/Transfer.R")
source("Core/Logit.R")


AVGMMIRemote = function(D,mvar,port,cent_host,cent_port)
{
  miss = is.na(D[,mvar])
  X = D[!miss,-mvar]
  y = D[!miss,mvar]
  
  while (TRUE)
  {
    rcon <- socketConnection(host="localhost",port=port,blocking=TRUE,server=TRUE,open="w+b",timeout=60*10)
    Sys.sleep(0.1)
    method = readBin(rcon,character())
    
    wcon <- socketConnection(cent_host,cent_port,open="w+b")
    
    if ( method == "Gaussian" )
      AVGMRemoteLS(X,y,wcon)
    else if ( method == "logistic" )
      AVGMRemoteLogit(X,y,wcon)
    
    close(rcon)
    close(wcon)
  }
}




AVGMRemoteLS = function(X,y,wcon,lam=1e-3)
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
  
  writeVec(beta,wcon)
  writeMat(iFisher,wcon)
  writeVec(SSE,wcon)
  writeVec(n,wcon)
}

AVGMRemoteLogit = function(X,y,wcon,lam=1e-3,maxiter=100)
{
  p = ncol(X)
  n = nrow(X)
  
  fit = Logit(X,y,lam=lam,maxiter=maxiter)
  
  writeVec(fit$beta,wcon)
  writeMat(fit$H,wcon)
  writeVec(n,wcon)
}

