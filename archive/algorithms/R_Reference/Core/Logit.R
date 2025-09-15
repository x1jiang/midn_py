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
