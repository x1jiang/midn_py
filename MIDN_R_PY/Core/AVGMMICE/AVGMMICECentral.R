# Function AVGMMICECentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: a vector of indices of missing variables
# type: a vector of "Gaussian" or "logistic" depending on missing variable types
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")
source("Core/LS.R")
source("Core/Logit.R")


AVGMMICECentral = function(D,M,mvar,type,iter,iter0,hosts,ports,cent_ports)
{
  D = cbind(D,1)
  p = ncol(D)
  n = nrow(D)
  miss = is.na(D)
  l = length(mvar)
  
  # Initialize
  for ( j in mvar )
  {
    idx1 = which(miss[,j])
    idx0 = which(!miss[,j])
    D[idx1,j] = mean(D[idx0,j])
  }

  K = length(hosts)
  wcons = list()
  rcons = list()
  for ( k in 1:K )
  {
    wcons[[k]] <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("Initialize",wcons[[k]])
    writeVec(mvar,wcons[[k]])
    rcons[[k]] <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
  }
  
  imp = list()
  for ( m in 1:M )
  {
    for ( it in 1:iter0 )
    {
      for ( i in 1:l )
      {
        j = mvar[i]
        midx = which(miss[,j])
        cidx = which(!miss[,j])
        nmidx = length(midx)
        if ( nmidx > 0 )
        {
          if ( nmidx>0 & type[i] == "Gaussian" )
          {
            fit.imp = AVGMLSNet(D,cidx,j,rcons=rcons,wcons=wcons)
            sig = sqrt(1/rgamma(1,(fit.imp$N+1)/2,(fit.imp$SSE+1)/2))
            alpha = fit.imp$beta + sig * t(chol(fit.imp$vcov))%*%rnorm(p-1)
            D[midx,j] = D[midx,-j] %*% alpha + rnorm(nmidx,0,sig)
            
            ImputeLS(j,alpha,sig,wcons)
          }
          else if ( type[i] == "logistic" )
          {
            fit.imp = AVGMLogitNet(D,cidx,j,rcons=rcons,wcons=wcons)

            alpha = fit.imp$beta + t(chol(fit.imp$vcov))%*%rnorm(p-1)
            pr = 1 / (1 + exp(-D[midx,-j] %*% alpha))
            D[midx,j] = rbinom(nmidx,1,pr)
            
            ImputeLogit(j,alpha,wcons)
          }
        }
      }
    }
    
    imp[[m]] = D
    iter0 = iter
  }

  for ( k in 1:K )
  {
    writeBin("End",wcons[[k]])
    close(wcons[[k]])
    close(rcons[[k]])
  }
  
  imp
}

