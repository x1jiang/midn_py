# Generate enhanced binary test data for SIMICE with 9 predictors

# Increased sample size for better numerical stability
n <- 30000  # larger sample size (increased from 3000)
cat("Generating dataset with", n, "observations for improved numerical stability\n")

# Define data split sizes
central_size <- floor(n / 6)      # 5,000 observations
remote1_size <- floor(n / 3)      # 10,000 observations  
remote2_size <- n - central_size - remote1_size  # 15,000 observations

cat("Data split sizes:\n")
cat("  Central site:", central_size, "observations\n")
cat("  Remote site 1:", remote1_size, "observations\n") 
cat("  Remote site 2:", remote2_size, "observations\n")

# Define block-specific parameters for each site
block_params <- list(
  central = list(
    # Coefficients for x1 generation (x2-x5) - SIGNIFICANTLY DIFFERENT from remotes
    # Strong relationship with x3 and x5, weak with x2 and x4
    x1_coef = c(0.05, 0.65, 0.05, 0.45),
    
    # Coefficients for x7 generation (x1, x2, x3, x6) - SIGNIFICANTLY DIFFERENT 
    # Much stronger dependence on x1 than remotes
    x7_coef = c(0.8, 0.1, 0.05, 0.05),
    
    # Coefficients for y generation (x1-x9) - SIGNIFICANTLY DIFFERENT pattern
    # Much stronger dependence on x1, x5 and x8 than remotes
    y_coef = c(0.5, 0.05, 0.05, 0.05, 0.35, 0.05, 0.05, 0.25, 0.05),
    
    # Missingness rates
    x1_miss_rate = 0.70,  # High missing rate
    x7_miss_rate = 0.60,  # High missing rate
    
    # Noise scaling - higher noise in central
    noise_scale = 0.4
  ),
  remote1 = list(
    # More balanced coefficients with stronger x2 influence
    x1_coef = c(0.3, 0.1, 0.25, 0.15),
    x7_coef = c(0.25, 0.25, 0.15, 0.15),
    y_coef = c(0.25, 0.25, 0.15, 0.15, 0.08, 0.12, 0.1, 0.08, 0.07),
    x1_miss_rate = 0.25,
    x7_miss_rate = 0.10,
    noise_scale = 0.25
  ),
  remote2 = list(
    # More balanced coefficients with stronger x3 influence
    x1_coef = c(0.15, 0.25, 0.15, 0.2),
    x7_coef = c(0.35, 0.15, 0.25, 0.1),
    y_coef = c(0.2, 0.15, 0.25, 0.1, 0.15, 0.05, 0.2, 0.05, 0.1),
    x1_miss_rate = 0.15,
    x7_miss_rate = 0.20,
    noise_scale = 0.3
  )
)

# Initialize data structures to hold generated data
dat <- data.frame(matrix(NA, nrow = n, ncol = 10))
names(dat) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y")

dat_truth <- data.frame(matrix(NA, nrow = n, ncol = 10))
names(dat_truth) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "y")

# Generate data for each block separately
start_idx <- 1
block_sizes <- c(central_size, remote1_size, remote2_size)
block_names <- c("central", "remote1", "remote2")

for (i in 1:3) {
  block_size <- block_sizes[i]
  block_name <- block_names[i]
  params <- block_params[[block_name]]
  
  # Set seed for reproducibility but make it different for each block
  set.seed(42 + i * 100)
  
  # End index for current block
  end_idx <- start_idx + block_size - 1
  
  # 1) Generate core predictors with correlation structure
  
  # Generate x2-x6, x8-x9 as independent variables
  x2 <- rnorm(block_size)  # standard normal
  x3 <- rnorm(block_size)  # standard normal
  x4 <- rnorm(block_size)  # standard normal
  x5 <- rnorm(block_size)  # standard normal
  x6 <- rnorm(block_size)  # standard normal
  x8 <- rnorm(block_size)  # standard normal
  x9 <- rnorm(block_size)  # standard normal
  
  # Generate x1 with block-specific coefficients
  x1_linear <- params$x1_coef[1] * x2 + params$x1_coef[2] * x3 + 
               params$x1_coef[3] * x4 + params$x1_coef[4] * x5 + 
               rnorm(block_size, mean = 0, sd = 0.5)
  
  # Convert x1 to binary (0/1)
  p_x1 <- plogis(x1_linear)
  x1_true <- rbinom(block_size, 1, p_x1)
  
  # Generate x7 with block-specific coefficients
  x7_linear <- params$x7_coef[1] * x1_true + params$x7_coef[2] * x2 + 
               params$x7_coef[3] * x3 + params$x7_coef[4] * x6 + 
               rnorm(block_size, mean = 0, sd = 0.5)
  
  # Impose random missingness on x7
  set.seed(123 + i * 10)
  p_miss_x7 <- params$x7_miss_rate
  is_miss_x7 <- rbinom(block_size, 1, p_miss_x7) == 1
  x7 <- x7_linear
  x7[is_miss_x7] <- NA
  
  # 2) Generate binary outcome y with block-specific coefficients
  lp_y <- -0.2 + 
          params$y_coef[1] * x1_true + 
          params$y_coef[2] * x2 + 
          params$y_coef[3] * x3 + 
          params$y_coef[4] * x4 + 
          params$y_coef[5] * x5 + 
          params$y_coef[6] * x6 + 
          params$y_coef[7] * x7_linear + 
          params$y_coef[8] * x8 + 
          params$y_coef[9] * x9
  
  # Add controlled noise with block-specific noise level
  signal_strength <- sd(lp_y)
  noise <- rnorm(block_size, mean = 0, sd = params$noise_scale * signal_strength)
  lp_y <- lp_y + noise
  
  # Scale the linear predictor to avoid extreme probabilities
  lp_y <- lp_y / (2 * sd(lp_y))
  
  # Convert to probabilities and generate binary outcome
  p_y <- plogis(lp_y)
  y <- rbinom(block_size, 1, p_y)
  
  # Sanity check for binary values
  if(any(y < 0 | y > 1 | !y %in% c(0, 1))) {
    cat("WARNING: Found non-binary values in y for", block_name, ". Fixing...\n")
    y <- as.integer(y > 0.5)
  }
  
  # 3) Create Missing At Random (MAR) pattern for x1
  # Different MAR mechanism for central vs remote sites
  if (block_name == "central") {
    # Central site: missingness strongly related to x3, x4, y
    lp_miss <- 0.1 * x2 + 0.4 * x3 + 0.3 * x4 + 0.3 * y
  } else {
    # Remote sites: standard pattern
    lp_miss <- 0.2 * x2 + 0.2 * x3 + 0.2 * y
  }
  
  # Scale to prevent extreme probabilities
  lp_miss_scaled <- lp_miss / (2 * sd(lp_miss))
  
  # Set intercept to achieve block-specific missingness rate
  target_miss_rate <- params$x1_miss_rate
  intercept <- qlogis(target_miss_rate) - mean(lp_miss_scaled)
  p_miss <- plogis(intercept + lp_miss_scaled)
  
  # Constrain probabilities to reasonable range (adjusted for central site)
  if (block_name == "central") {
    p_miss <- pmin(pmax(p_miss, 0.60), 0.85)  # Higher range for central site [60%, 85%]
  } else {
    p_miss <- pmin(pmax(p_miss, 0.05), 0.35)  # Original range for remote sites [5%, 35%]
  }
  
  # Generate missing indicators and apply to x1
  set.seed(456 + i * 10)
  is_miss <- rbinom(block_size, 1, p_miss) == 1
  x1 <- x1_true
  x1[is_miss] <- NA
  
  # Store data in the combined dataset
  dat[start_idx:end_idx, ] <- data.frame(
    x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5,
    x6 = x6, x7 = x7, x8 = x8, x9 = x9, y = y
  )
  
  # For the truth dataset, ensure NO missing values
  dat_truth[start_idx:end_idx, ] <- data.frame(
    x1 = x1_true, x2 = x2, x3 = x3, x4 = x4, x5 = x5,
    x6 = x6, x7 = x7_linear, x8 = x8, x9 = x9, y = y
  )
  
  # Report block-specific missingness for both X1 and X7
  cat(block_name, "site missingness rates:", 
      "X1:", round(100 * mean(is.na(x1)), 1), "%,",
      "X7:", round(100 * mean(is.na(x7)), 1), "%\n")
  
  # Update start index for next block
  start_idx <- end_idx + 1
}

cat("Data split sizes:\n")
cat("  Central site:", central_size, "observations\n")
cat("  Remote site 1:", remote1_size, "observations\n") 
cat("  Remote site 2:", remote2_size, "observations\n")

# Check missingness distribution across sites
central_end <- central_size
remote1_end <- central_size + remote1_size
central_miss_rate_x1 <- mean(is.na(dat$x1[1:central_end]))
remote1_miss_rate_x1 <- mean(is.na(dat$x1[(central_end + 1):remote1_end]))
remote2_miss_rate_x1 <- mean(is.na(dat$x1[(remote1_end + 1):n]))

central_miss_rate_x7 <- mean(is.na(dat$x7[1:central_end]))
remote1_miss_rate_x7 <- mean(is.na(dat$x7[(central_end + 1):remote1_end]))
remote2_miss_rate_x7 <- mean(is.na(dat$x7[(remote1_end + 1):n]))

cat("Missing rates by site (X1):\n")
cat("  Central:", round(100 * central_miss_rate_x1, 1), "%\n")
cat("  Remote1:", round(100 * remote1_miss_rate_x1, 1), "%\n") 
cat("  Remote2:", round(100 * remote2_miss_rate_x1, 1), "%\n")

cat("Missing rates by site (X7):\n")
cat("  Central:", round(100 * central_miss_rate_x7, 1), "%\n")
cat("  Remote1:", round(100 * remote1_miss_rate_x7, 1), "%\n") 
cat("  Remote2:", round(100 * remote2_miss_rate_x7, 1), "%\n")

# Verify truth dataset has no missing values
cat("\nChecking truth dataset for missing values:\n")
cat("  Missing values in truth X1:", sum(is.na(dat_truth$x1)), "\n")
cat("  Missing values in truth X7:", sum(is.na(dat_truth$x7)), "\n")

# Create ground truth dataset (already initialized and filled in the for loop above)

# 5) Summary statistics and validation
cat("\n=== Data Generation Summary ===\n")
cat("Total observations:", n, "\n")
cat("Missing in x1:", sum(is.na(dat$x1)), "(", round(100*mean(is.na(dat$x1)), 1), "%)\n")
cat("Missing in x7:", sum(is.na(dat$x7)), "(", round(100*mean(is.na(dat$x7)), 1), "%)\n")

# Verify no missing values in truth dataset again
if(sum(is.na(dat_truth)) > 0) {
  cat("WARNING: Truth dataset contains", sum(is.na(dat_truth)), "missing values!\n")
} else {
  cat("GOOD: Truth dataset contains no missing values, as required.\n")
}

# Check correlation matrix of complete variables
cat("\nCorrelation matrix (complete variables only):\n")
complete_vars <- dat_truth[, c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9")]
print(round(cor(complete_vars), 2))

# Display first few rows for inspection
cat("\nFirst 6 rows of dataset:\n")
print(head(dat))

# 6) Split data into sites and save files

# Central site data (first 1/6 of observations)
central_data <- dat[1:central_size, ]
central_data_truth <- dat_truth[1:central_size, ]

write.csv(central_data, file = "./multi_missing_central.csv", row.names = FALSE)
write.csv(central_data_truth, file = "./multi_missing_central_truth.csv", row.names = FALSE)

# Remote site 1 data (next 1/3 of observations)
remote1_start <- central_size + 1
remote1_end <- central_size + remote1_size
remote1_data <- dat[remote1_start:remote1_end, ]
remote1_data_truth <- dat_truth[remote1_start:remote1_end, ]

write.csv(remote1_data, file = "./multi_missing_remote1.csv", row.names = FALSE)
write.csv(remote1_data_truth, file = "./multi_missing_remote1_truth.csv", row.names = FALSE)

# Remote site 2 data (remaining 1/2 of observations)
remote2_start <- remote1_end + 1
remote2_data <- dat[remote2_start:n, ]
remote2_data_truth <- dat_truth[remote2_start:n, ]

write.csv(remote2_data, file = "./multi_missing_remote2.csv", row.names = FALSE)
write.csv(remote2_data_truth, file = "./multi_missing_remote2_truth.csv", row.names = FALSE)

# 7) Final validation and reporting
cat("\n=== File Generation Summary ===\n")
cat("Generated files:\n")
cat("  Central site: multi_missing_central.csv (", nrow(central_data), " rows)\n")
cat("  Central truth: multi_missing_central_truth.csv (", nrow(central_data_truth), " rows)\n")
cat("  Remote site 1: multi_missing_remote1.csv (", nrow(remote1_data), " rows)\n")
cat("  Remote site 1 truth: multi_missing_remote1_truth.csv (", nrow(remote1_data_truth), " rows)\n")
cat("  Remote site 2: multi_missing_remote2.csv (", nrow(remote2_data), " rows)\n") 
cat("  Remote site 2 truth: multi_missing_remote2_truth.csv (", nrow(remote2_data_truth), " rows)\n")

# Final missing data distribution
central_x1_miss <- sum(is.na(central_data$x1))
remote1_x1_miss <- sum(is.na(remote1_data$x1))
remote2_x1_miss <- sum(is.na(remote2_data$x1))
total_x1_miss <- central_x1_miss + remote1_x1_miss + remote2_x1_miss

central_x7_miss <- sum(is.na(central_data$x7))
remote1_x7_miss <- sum(is.na(remote1_data$x7))
remote2_x7_miss <- sum(is.na(remote2_data$x7))
total_x7_miss <- central_x7_miss + remote1_x7_miss + remote2_x7_miss

cat("\n=== Missing Value Distribution (X1) ===\n")
cat("Central site:", central_x1_miss, "missing (", 
    round(100 * central_x1_miss / nrow(central_data), 1), "% of central,",
    round(100 * central_x1_miss / total_x1_miss, 1), "% of all missing)\n")
cat("Remote site 1:", remote1_x1_miss, "missing (",
    round(100 * remote1_x1_miss / nrow(remote1_data), 1), "% of remote1,",
    round(100 * remote1_x1_miss / total_x1_miss, 1), "% of all missing)\n")
cat("Remote site 2:", remote2_x1_miss, "missing (",
    round(100 * remote2_x1_miss / nrow(remote2_data), 1), "% of remote2,",
    round(100 * remote2_x1_miss / total_x1_miss, 1), "% of all missing)\n")

cat("\n=== Missing Value Distribution (X7) ===\n")
cat("Central site:", central_x7_miss, "missing (", 
    round(100 * central_x7_miss / nrow(central_data), 1), "% of central,",
    round(100 * central_x7_miss / total_x7_miss, 1), "% of all missing)\n")
cat("Remote site 1:", remote1_x7_miss, "missing (",
    round(100 * remote1_x7_miss / nrow(remote1_data), 1), "% of remote1,",
    round(100 * remote1_x7_miss / total_x7_miss, 1), "% of all missing)\n")
cat("Remote site 2:", remote2_x7_miss, "missing (",
    round(100 * remote2_x7_miss / nrow(remote2_data), 1), "% of remote2,",
    round(100 * remote2_x7_miss / total_x7_miss, 1), "% of all missing)\n")

# Final check of truth files for missing values
central_truth_missing <- sum(is.na(central_data_truth))
remote1_truth_missing <- sum(is.na(remote1_data_truth))
remote2_truth_missing <- sum(is.na(remote2_data_truth))

if(central_truth_missing > 0 || remote1_truth_missing > 0 || remote2_truth_missing > 0) {
  cat("\nWARNING: Truth files contain missing values:\n")
  cat("  Central truth missing:", central_truth_missing, "\n")
  cat("  Remote1 truth missing:", remote1_truth_missing, "\n")
  cat("  Remote2 truth missing:", remote2_truth_missing, "\n")
} else {
  cat("\nGOOD: All truth files are complete with no missing values.\n")
}

cat("\nData generation completed successfully!\n")
