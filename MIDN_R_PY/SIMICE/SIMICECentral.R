# Function SIMICECentral
# Arguments:
# D: Data matrix
# M: Number of imputations
# mvar: a vector of indices of missing variables
# type: a vector of "Gaussian" or "logistic" depending on missing variable types
# iter: the number of imputation iterations between extracted imputed data
# iter0: the number of imputation iterations before the first extracted imputed data
# hosts: a vector of hostnames of remote sites
# ports: a vector of ports of remote sites
# cent_ports: a vector of local listening ports dedicated to corresponding remote sites


source("Core/Transfer.R")
source("Core/LS.R")
source("Core/Logit.R")

# Set to TRUE to enable detailed debug information, similar to Python version
print_debug_info = TRUE

SIMICECentral = function(D,M,mvar,type,iter,iter0,hosts,ports,cent_ports)
{
  # Create debug counters similar to Python implementation
  debug_counters <- list(
    ls_calls = 0,
    logit_calls = 0,
    impute_ls_calls = 0,
    impute_logit_calls = 0,
    iteration = 0,
    imputation = 0
  )

  start_time <- Sys.time()
  
  D = cbind(D,1)
  p = ncol(D)
  n = nrow(D)
  miss = is.na(D)
  l = length(mvar)
  
  if (print_debug_info) {
    cat(sprintf("[CENTRAL] Debug mode ENABLED - detailed logging will be shown\n"))
    cat(sprintf("[CENTRAL] Starting SIMICE with M=%d, iter=%d, iter0=%d\n", M, iter, iter0))
    cat(sprintf("[CENTRAL] Data shape: (%d, %d), mvar: %s\n", 
                nrow(D)-1, ncol(D)-1, paste("[", paste(mvar, collapse=", "), "]", sep="")))
    cat(sprintf("[CENTRAL] Variable types: %s\n", paste(paste("'", type, "'", sep=""), collapse=", ")))
    cat(sprintf("[CENTRAL] Expected remote sites: %s\n", paste(hosts, collapse=", ")))
    
    # Display columns with NaN values
    nan_cols <- which(colSums(is.na(D)) > 0)
    cat(sprintf("[CENTRAL] Columns with NaN: %s\n", paste(nan_cols, collapse=", ")))
    
    for (j in mvar) {
      nan_count <- sum(is.na(D[,j]))
      nan_percent <- 100 * nan_count / nrow(D)
      cat(sprintf("[CENTRAL] Column %d: %d NaN values (%.1f%%)\n", j, nan_count, nan_percent))
    }
  }
  
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
  
  if (print_debug_info) {
    cat(sprintf("Initializing remote sites: %s\n", paste(hosts, collapse=", ")))
  }
  
  for ( k in 1:K )
  {
    if (print_debug_info) {
      cat(sprintf("Connecting to remote site %s at port %d\n", hosts[k], ports[k]))
    }
    
    wcons[[k]] <- socketConnection(hosts[k],ports[k],open="w+b")
    writeBin("Initialize",wcons[[k]])
    writeVec(mvar,wcons[[k]])
    rcons[[k]] <- socketConnection(host="localhost",port=cent_ports[k],blocking=TRUE,server=TRUE,open="w+b")
    
    if (print_debug_info) {
      cat(sprintf("Initialized remote site %s with mvar %s\n", 
                 hosts[k], paste("[", paste(mvar, collapse=", "), "]", sep="")))
    }
  }
  
  imp = list()
  for ( m in 1:M )
  {
    debug_counters$imputation = m
    
    if (print_debug_info) {
      cat(sprintf("[CENTRAL][IMPUTATION #%d] Starting imputation %d/%d with %d iterations\n", 
                  m, m, M, iter0))
      imputation_start_time <- Sys.time()
    }
    
    for ( it in 1:iter0 )
    {
      debug_counters$iteration = debug_counters$iteration + 1
      iteration_num = debug_counters$iteration
      
      if (print_debug_info) {
        cat(sprintf("[CENTRAL][ITERATION #%d] Starting iteration %d/%d\n", 
                    iteration_num, it, iter0))
        iteration_start_time <- Sys.time()
      }
      
      for ( i in 1:l )
      {
        j = mvar[i]
        midx = which(miss[,j])
        cidx = which(!miss[,j])
        nmidx = length(midx)
        
        if (print_debug_info) {
          cat(sprintf("[CENTRAL][VAR j=%d] Processing variable %d/%d, type=%s\n", 
                      j, i, l, type[i]))
          cat(sprintf("[CENTRAL][VAR j=%d] Missing: %d, Non-missing: %d\n", 
                      j, nmidx, length(cidx)))
          var_start_time <- Sys.time()
        }
        
        if ( nmidx > 0 )
        {
          if ( nmidx>0 & type[i] == "Gaussian" )
          {
            debug_counters$ls_calls = debug_counters$ls_calls + 1
            ls_iter = debug_counters$ls_calls
            
            if (print_debug_info) {
              cat(sprintf("[CENTRAL][LS #%d] Running linear regression for variable %d\n", 
                          ls_iter, j))
            }
            
            fit.imp = SILSNet(D,cidx,j,rcons=rcons,wcons=wcons)
            
            sig = sqrt(1/rgamma(1,(fit.imp$N+1)/2,(fit.imp$SSE+1)/2))
            alpha = fit.imp$beta + sig * backsolve(fit.imp$cgram,rnorm(p-1))
            D[midx,j] = D[midx,-j] %*% alpha + rnorm(nmidx,0,sig)
            
            debug_counters$impute_ls_calls = debug_counters$impute_ls_calls + 1
            
            if (print_debug_info) {
              cat(sprintf("[CENTRAL][LS #%d] Imputing with alpha range=[%.4f, %.4f], sigma=%.4f\n", 
                          ls_iter, min(alpha), max(alpha), sig))
            }
            
            ImputeLS(j,alpha,sig,wcons)
          }
          else if ( type[i] == "logistic" )
          {
            debug_counters$logit_calls = debug_counters$logit_calls + 1
            logit_iter = debug_counters$logit_calls
            
            if (print_debug_info) {
              cat(sprintf("[CENTRAL][LOGIT #%d] Running logistic regression for variable %d\n", 
                          logit_iter, j))
            }
            
            fit.imp = SILogitNet(D,cidx,j,rcons=rcons,wcons=wcons)
            
            cH = chol(fit.imp$H)
            alpha = fit.imp$beta + backsolve(cH,rnorm(p))
            pr = 1 / (1 + exp(-D[midx,-j] %*% alpha))
            D[midx,j] = rbinom(nmidx,1,pr)
            
            debug_counters$impute_logit_calls = debug_counters$impute_logit_calls + 1
            
            if (print_debug_info) {
              cat(sprintf("[CENTRAL][LOGIT #%d] Imputing with alpha range=[%.4f, %.4f]\n", 
                          logit_iter, min(alpha), max(alpha)))
            }
            
            ImputeLogit(j,alpha,wcons)
          }
        }
        
        if (print_debug_info) {
          var_end_time <- Sys.time()
          var_duration <- as.numeric(difftime(var_end_time, var_start_time, units="secs"))
          cat(sprintf("[CENTRAL][VAR j=%d] Variable processed in %.3fs\n", j, var_duration))
        }
      }
      
      if (print_debug_info) {
        iteration_end_time <- Sys.time()
        iteration_duration <- as.numeric(difftime(iteration_end_time, iteration_start_time, units="secs"))
        cat(sprintf("[CENTRAL][ITERATION #%d] Iteration %d/%d completed in %.3fs\n", 
                   iteration_num, it, iter0, iteration_duration))
      }
    }
    
    imp[[m]] = D
    iter0 = iter
    
    if (print_debug_info) {
      imputation_end_time <- Sys.time()
      imputation_duration <- as.numeric(difftime(imputation_end_time, imputation_start_time, units="secs"))
      cat(sprintf("[CENTRAL][IMPUTATION #%d] Imputation %d/%d completed in %.3fs\n", 
                 m, m, M, imputation_duration))
      
      # Log summary statistics for imputed variables
      for (i in 1:l) {
        j = mvar[i]
        if (type[i] == "Gaussian") {
          cat(sprintf("[CENTRAL][IMPUTATION #%d] Variable j=%d stats: min=%.4f, max=%.4f, mean=%.4f\n",
                     m, j, min(D[,j]), max(D[,j]), mean(D[,j])))
        } else if (type[i] == "logistic") {
          counts = table(D[,j])
          cat(sprintf("[CENTRAL][IMPUTATION #%d] Variable j=%d counts: 0=%d, 1=%d\n",
                     m, j, counts["0"], counts["1"]))
        }
      }
    }
  }

  for ( k in 1:K )
  {
    writeBin("End",wcons[[k]])
    close(wcons[[k]])
    close(rcons[[k]])
    
    if (print_debug_info) {
      cat(sprintf("[CENTRAL] Closed connection with remote site %s\n", hosts[k]))
    }
  }
  
  if (print_debug_info) {
    end_time <- Sys.time()
    total_duration <- as.numeric(difftime(end_time, start_time, units="secs"))
    cat(sprintf("[CENTRAL] SIMICE completed: %d imputations in %.3fs\n", M, total_duration))
    cat(sprintf("[CENTRAL] Debug counters: ls_calls=%d, logit_calls=%d, impute_ls_calls=%d, impute_logit_calls=%d, iteration=%d, imputation=%d\n", 
                debug_counters$ls_calls, debug_counters$logit_calls, 
                debug_counters$impute_ls_calls, debug_counters$impute_logit_calls,
                debug_counters$iteration, debug_counters$imputation))
  }
  
  imp
}

