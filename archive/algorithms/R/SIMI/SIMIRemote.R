# Function SIMIRemote
# Arguments:
# D: Data matrix
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site


source("Core/Transfer.R")


SIMIRemote = function(D,mvar,port,cent_host,cent_port)
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
      SIRemoteLS(X,y,wcon)
    else if ( method == "logistic" )
      SIRemoteLogit(X,y,rcon,wcon)
    
    close(rcon)
    close(wcon)
  }
}




SIRemoteLS = function(X,y,wcon)
{
  p = ncol(X)
  n = nrow(X)
  XX = t(X)%*%X
  Xy = drop(t(X)%*%y)
  yy = sum(y^2)
  
  writeVec(n,wcon)
  writeMat(XX,wcon)
  writeVec(Xy,wcon)
  writeVec(yy,wcon)
}

SIRemoteLogit = function(X,y,rcon,wcon)
{
  p = ncol(X)
  n = nrow(X)
  
  writeVec(n,wcon)
  
  while (TRUE)
  {
    mode = readBin(rcon,integer())
    if ( mode == 0 )
      break
    else
    {
      beta = readVec(rcon)
      xb = drop(X%*%beta)
      pr = 1/(1+exp(-xb))
      Q = sum(y*xb) + sum(log(1-pr[pr<0.5])) + sum(log(pr[pr>=0.5])-xb[pr>=0.5])
      if ( mode == 1 )
      {
        H = t(X)%*%(X*pr*(1-pr))
        writeMat(H,wcon)
        g = t(X)%*%(y-pr)
        writeVec(g,wcon)
      }
      writeVec(Q,wcon)
    }
  }
}
