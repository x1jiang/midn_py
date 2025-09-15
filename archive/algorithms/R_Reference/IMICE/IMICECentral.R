# Function IMICECentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type

source("Core/LS.R")
source("Core/Logit.R")

IMICECentral = function(D,M,mvar,method,iter,iter0)
{
  D = cbind(D,1)
  p = ncol(D)
  n = nrow(D)
  miss = is.na(D)
  l = length(mvar)

  for ( j in mvar )
  {
    idx1 = which(miss[,j])
    idx0 = which(!miss[,j])
    D[idx1,j] = mean(D[idx0,j])
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
        nmidx = length(midx)
        
        if ( nmidx > 0 )
        {
          Xidx = setdiff(1:p,j)
          
          if ( nmidx>0 & method[i] == "Gaussian" )
          {
            fit.imp = LS(D[-midx,Xidx],D[-midx,j])
            sig = sqrt(1/rgamma(1,(fit.imp$n+1)/2,(fit.imp$SSE+1)/2))
            alpha = fit.imp$beta + sig * backsolve(fit.imp$cgram,rnorm(p))
            D[midx,j] = D[midx,Xidx] %*% alpha + rnorm(nmidx,0,sig)
          }
          else if ( method[i] == "logistic" )
          {
            fit.imp = Logit(D[-midx,Xidx],D[-midx,j])
            cH = chol(fit.imp$H)
            alpha = fit.imp$beta + backsolve(cH,rnorm(p))
            pr = 1 / (1 + exp(-D[midx,Xidx] %*% alpha))
            D[midx,j] = rbinom(nmidx,1,pr)
          }
        }
      }
    }
      
    imp[[m]] = D
    iter0 = iter
  }
  
  imp
}


