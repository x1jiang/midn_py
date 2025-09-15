# Function CSLMICERemote
# Arguments:
# D: Data matrix
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site


source("Core/Transfer.R")


CSLMICERemote = function(D,port,cent_host,cent_port)
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
          CSLRemoteLS(X,y,rcon,wcon)
        else if ( method == "logistic" )
          CSLRemoteLogit(X,y,rcon,wcon)
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


CSLRemoteLS = function(X,y,rcon,wcon)
{
  n = nrow(X)
  writeVec(n,wcon)
  
  betabar = readVec(rcon)
  g = -drop(t(X)%*%(y-X%*%betabar))/n 
  writeVec(g,wcon)
}


CSLRemoteLogit = function(X,y,rcon,wcon)
{
  n = nrow(X)
  writeVec(n,wcon)
  
  betabar = readVec(rcon)
  xb = drop(X%*%betabar)
  pr = 1/(1+exp(-xb))
  g = drop(t(X)%*%(y-pr))/n
  writeVec(g,wcon)
}
