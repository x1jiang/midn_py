# Function HDMICentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: Index of missing variable
# method: "Gaussian" or "logistic" depending on missing data type
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")



HDMICentral <- function(D, M, mvar, method, hosts, ports, cent_ports) {
  n <- nrow(D)
  p <- ncol(D) - 1
  miss <- is.na(D[, mvar])
  nm <- sum(miss)
  nc <- n - nm

  X <- D[, -mvar]
  y <- D[, mvar]

  if (method == "Gaussian") {
    HD <- HDMICentralLS(X, y, hosts, ports, cent_ports)
    # cvcov = chol(HD$vcov)
  } else if (method == "logistic") {
    HD <- HDMICentralLogit(X, y, hosts, ports, cent_ports)
    # cvcov = chol(HD$vcov)
  }

  imp <- NULL
  for (m in 1:M)
  {
    if (method == "Gaussian") {
      alpha <- mvtnorm::rmvnorm(n = 1,
                                mean = HD$beta,
                                sigma = HD$vcov)
      # the variable used in selection but not in outcome model is put at last, the p-th variable in X
      rho <- alpha[2 * p + 1]
      rho <- pmax(pmin(rho, 100), -100)
      rho_star <- (exp(2 * rho) - 1)/(1 + exp(2 * rho))

      sigma <- alpha[2 * p]
      sigma_star <- exp(sigma)

      var_sel <- X[miss, ]
      var_out <- var_sel[, -p]

      betaxstar_sel <- c(var_sel %*% alpha[1:p])
      betaxstar_out <- c(var_out %*% alpha[(p+1):(2*p-1)])

      D[miss, mvar] <- betaxstar_out +
                       sigma_star*rho_star*
                       (-dnorm(betaxstar_sel)/(pnorm(-betaxstar_sel))) +
                       rnorm(nm, 0, sigma_star)

    } else if (method == "logistic") {
      alpha <- mvtnorm::rmvnorm(n = 1,
                                mean = HD$beta,
                                sigma = HD$vcov)
      # the variable used in selection but not in outcome model is put at last, the p-th variable in X
      rho <- alpha[2 * p]
      rho <- pmax(pmin(rho, 100), -100)
      rho_star <- (exp(2 * rho) - 1)/(1 + exp(2 * rho))

      var_sel <- X[miss, ]
      var_out <- var_sel[, -p]

      betaxstar_sel <- c(var_sel %*% alpha[1:p])
      betaxstar_out <- c(var_out %*% alpha[(p+1):(2*p-1)])

      p.star <- pbivnorm::pbivnorm(betaxstar_out, -betaxstar_sel, -rho_star)/pnorm(-betaxstar_sel)
      p.star <- pmax(pmin(p.star, 1-1e-6), 1e-6)

      D[miss, mvar] <- rbinom(nm, 1, p.star)
    }

    imp[[m]] <- D
  }

  imp
}

HDMICentralLS <- function(X, y, hosts, ports, cent_ports, lam = 1e-3) {
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

  HD <- list()
  HD$beta <- fit$coefficients * n
  HD$vcov <- fit$Vb * n^2
  HD$n <- n

  K <- length(hosts)
  for (k in 1:K)
  {
    rcon <- socketConnection(hosts[k], ports[k], open = "w+b")
    writeBin("Gaussian", rcon)
    wcon <- socketConnection(host = "localhost", port = cent_ports[k], blocking = TRUE, server = TRUE, open = "w+b")

    beta <- readVec(wcon)
    vcov <- readMat(wcon)
    # iFisher <- readMat(wcon)
    SSE <- readVec(wcon)
    n <- readVec(wcon)

    close(rcon)
    close(wcon)

    HD$beta <- HD$beta + beta * n
    HD$vcov <- HD$vcov + vcov * n^2
    # HD$SSE <- HD$SSE + SSE
    HD$n <- HD$n + n
  }
  HD$beta <- HD$beta / HD$n
  HD$vcov <- HD$vcov / HD$n^2

  HD
}

HDMICentralLogit <- function(X, y, hosts, ports, cent_ports, lam = 1e-3, maxiter = 100) {
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

  HD <- list()
  HD$beta <- fit$coefficients * n
  HD$vcov <- fit$Vb * n^2
  HD$n <- n

  K <- length(hosts)
  for (k in 1:K)
  {
    rcon <- socketConnection(hosts[k], ports[k], open = "w+b")
    writeBin("logistic", rcon)
    wcon <- socketConnection(host = "localhost", port = cent_ports[k], blocking = TRUE, server = TRUE, open = "w+b")

    beta <- readVec(wcon)
    vcov <- readMat(wcon)
    n <- readVec(wcon)

    close(rcon)
    close(wcon)

    HD$beta <- HD$beta + beta * n
    HD$vcov <- HD$vcov + vcov * n^2
    HD$n <- HD$n + n
  }
  HD$beta <- HD$beta / HD$n
  HD$vcov <- HD$vcov / HD$n^2

  HD
}
