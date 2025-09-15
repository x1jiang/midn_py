# Function CSLMIRemote
# Arguments:
# D: Data matrix
# mvar: Index of missing variable
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site

source("Core/Transfer.R")


CSLMIRemote = function(D,mvar,port,cent_host,cent_port)
{
  miss = is.na(D[,mvar])
  X = D[!miss,-mvar]
  y = D[!miss,mvar]

  n = nrow(X)
  p = ncol(X)
  
  while (TRUE)
  {
    rcon <- socketConnection(host="localhost",port=port,blocking=TRUE,server=TRUE,open="w+b",timeout=60*10)
    Sys.sleep(0.1)
    method = readBin(rcon,character())
    betabar = readBin(rcon,numeric(),p)
    
    if ( method == "Gaussian" )
      g = -drop(t(X)%*%(y-X%*%betabar))/n 
    else if ( method == "logistic" )
    {
      xb = drop(X%*%betabar)
      pr = 1/(1+exp(-xb))
      g = drop(t(X)%*%(y-pr))/n
    }

    wcon <- socketConnection(cent_host,cent_port,open="w+b")
    writeBin(as.integer(n),wcon)
    writeBin(g,wcon)
    
    close(rcon)
    close(wcon)
  }
}


