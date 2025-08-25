# Function HDMIRemote
# Arguments:
# D: Data matrix
# port: local listening port
# cent_host: hostname of central site
# cent_port: port of central site


source("Core/Transfer.R")


HDMIRemote = function(D,mvar,port,cent_host,cent_port)
{
  miss = is.na(D[,mvar])
  X = D[!miss,-mvar]
  y = D[!miss,mvar]
  
  while (TRUE)
  {
    rcon <- socketConnection(host="localhost",port=port,blocking=TRUE,server=TRUE,open="w+b")
    method = readBin(rcon,character())
    
    wcon <- socketConnection(cent_host,cent_port,open="w+b")
    
    if ( method == "Gaussian" )
      HDMIRemoteLS(X,y,wcon)
    else if ( method == "logistic" )
      HDMIRemoteLogit(X,y,wcon)
    
    close(rcon)
    close(wcon)
  }
}




HDMIRemoteLS = function(X,y,wcon,lam=1e-3)
{
  p <- ncol(X)
  n <- nrow(X)
  p_s <- p
  p_o <- p - 1

  dd <- data.frame(X, y)
  names(dd) <- c(paste0("x", 1:p), "y")
  dd$select <- !is.na(dd$y)

  # all variables are used in the selection model
  var_sel <- names(dd)[1:p]
  # the last variable is not used in the outcome model, per the exclusion-restriction rule
  var_out <- names(dd)[1:(p - 1)]

  formula_s <- as.formula(paste(
    "select ~",
    paste(var_sel, collapse = "+"), " - 1"
  ))
  formula_o <- as.formula(paste(
    "y ~",
    paste(var_out, collapse = "+"), " - 1"
  ))

  fit <- GJRM::copulaSampleSel(list(formula_s, formula_o),
    data = dd, fp = TRUE
  )

  beta <- fit$coefficients
  vcov <- fit$Vb
  SSE <- NULL
  
  writeVec(beta,wcon)
  writeMat(vcov,wcon)
  writeVec(SSE,wcon)
  writeVec(n,wcon)
}

HDMIRemoteLogit = function(X,y,wcon,lam=1e-3,maxiter=100)
{
  p <- ncol(X)
  n <- nrow(X)
  p_s <- p
  p_o <- p - 1

  dd <- data.frame(X, y)
  names(dd) <- c(paste0("x", 1:p), "y")
  dd$select <- !is.na(dd$y)

  # all variables are used in the selection model
  var_sel <- names(dd)[1:p]
  # the last variable is not used in the outcome model, per the exclusion-restriction rule
  var_out <- names(dd)[1:(p - 1)]

  formula_s <- as.formula(paste(
    "select ~",
    paste(var_sel, collapse = "+"), " - 1"
  ))
  formula_o <- as.formula(paste(
    "y ~",
    paste(var_out, collapse = "+"), " - 1"
  ))

  fit <- GJRM::SemiParBIV(list(formula_s, formula_o),
    data = dd, Model = "BSS",
    fp = TRUE
  )
  
  writeVec(fit$coefficients,wcon)
  writeMat(fit$Vb,wcon)
  writeVec(n,wcon)
}

