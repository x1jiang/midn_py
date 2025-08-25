# Function AVGMMICERemote
# Arguments:
# D: Data matrix
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site


source("Core/Transfer.R")
source("Core/Logit.R")

AVGMMICERemote = function(D,port,cent_host,cent_port)
{
  D = cbind(D,1)
  p = ncol(D)
  
  while (TRUE)
  {
    rcon <- socketConnection(host="localhost",port=port,blocking=TRUE,server=TRUE,open="w+b",timeout=60*10)
    Sys.sleep(0.1)
    wcon <- socketConnection(cent_host,cent_port,open="w+b")

    while ( TRUE )
    {
      inst = readBin(rcon,character())
      if ( inst == "Initialize" )
      {
        mvar = readVec(rcon)
        miss = is.na(D)
        idx = rowSums(miss[,-mvar]) == 0
        n = length(idx)
        DD = D[idx,]
        miss = is.na(DD)
        for ( j in mvar )
        {
          idx1 = which(miss[,j])
          idx0 = which(!miss[,j])
          DD[idx1,j] = mean(D[idx0,j])
        }
      }
      else if ( inst == "Information" )
      {
        method = readBin(rcon,character())
        yidx = readBin(rcon,integer())
        X = matrix(DD[!miss[,yidx],-yidx],ncol=p-1)
        y = DD[!miss[,yidx],yidx]
        if ( method == "Gaussian" )
          AVGMRemoteLS(X,y,wcon)
        else if ( method == "logistic" )
          AVGMRemoteLogit(X,y,wcon)
      }
      else if ( inst == "Impute" )
      {
        method = readBin(rcon,character())
        yidx = readBin(rcon,integer())
        midx = which(miss[,yidx])
        nmidx = length(midx)
        X = matrix(DD[midx,-yidx],ncol=p-1)
        if ( method == "Gaussian" )
        {
          beta = readBin(rcon,numeric(),p-1)
          sig = readBin(rcon,numeric())
          DD[midx,yidx] = X %*% beta + rnorm(nmidx,0,sig)
        }
        else if ( method == "logistic" )
        {
          alpha = readBin(rcon,numeric(),p-1)
          pr = 1 / (1 + exp(-X %*% alpha))
          DD[midx,yidx] = rbinom(nmidx,1,pr)
        }
      }
      else if ( inst == "End" )
        break
    }
    
    
    close(rcon)
    close(wcon)
  }
}


AVGMRemoteLS = function(X,y,wcon,lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)
  XX = t(X)%*%X + diag(n*lam,p)
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


AVGMRemoteLogit = function(X,y,wcon,lam=1e-3)
{
  p = ncol(X)
  n = nrow(X)
  
  fit = Logit(X,y,lam=lam)
  
  writeVec(fit$beta,wcon)
  writeMat(fit$H,wcon)
  writeVec(n,wcon)
}
