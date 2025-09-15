# Generate enhanced binary test data for SIMICE with 9 predictors

# Increased sample size for better numerical stability
n <- 30000  # larger sample size (increased from 3000)
cat("Generating dataset with", n, "observations for improved numerical stability\n")

set.seed(42)  # For reproducibility

# 1) Generate core predictors with correlation structure

# Generate x2-x6, x8-x9 as independent variables first
x2 <- rnorm(n)  # standard normal
x3 <- rnorm(n)  # standard normal
x4 <- rnorm(n)  # standard normal
x5 <- rnorm(n)  # standard normal
x6 <- rnorm(n)  # standard normal
x8 <- rnorm(n)  # standard normal
x9 <- rnorm(n)  # standard normal

# Generate x1 with linear relationship to x2-x5 (with Gaussian noise)
# This will be the variable that has missing values
x1_linear <- 0.2 * x2 + 0.2 * x3 + 0.2 * x4 + 0.2 * x5 + rnorm(n, mean = 0, sd = 0.5)

# Convert x1 to binary (0/1) - this ensures we have binary values before imposing missingness
p_x1 <- plogis(x1_linear)  # Convert to probability using logistic function
x1_true <- rbinom(n, 1, p_x1)  # Generate binary 0/1 values

# Generate x7 as a function of x1_true, x2, x3, x6 (with Gaussian noise)
x7_linear <- 0.3 * x1_true + 0.2 * x2 + 0.2 * x3 + 0.2 * x6 + rnorm(n, mean = 0, sd = 0.5)

# Impose random missingness on x7 (15% missing at random)
set.seed(123)  # For reproducibility of missingness pattern
p_miss_x7 <- 0.15
is_miss_x7 <- rbinom(n, 1, p_miss_x7) == 1
x7 <- x7_linear
x7[is_miss_x7] <- NA

# 2) Generate binary outcome y based on all predictors (use x7_linear, not x7 with missing)
# Use a combination of all variables with decreasing importance
lp_y <- -0.2 + 0.3 * x1_true + 0.25 * x2 + 0.2 * x3 + 0.15 * x4 + 0.1 * x5 + 
        0.1 * x6 + 0.15 * x7_linear + 0.1 * x8 + 0.05 * x9

# Add controlled noise to the linear predictor (20% of signal strength)
signal_strength <- sd(lp_y)
noise <- rnorm(n, mean = 0, sd = 0.2 * signal_strength)
lp_y <- lp_y + noise

# Scale the linear predictor to avoid extreme probabilities
lp_y <- lp_y / (2 * sd(lp_y))

# Convert to probabilities and generate binary outcome
p_y <- plogis(lp_y)
y <- rbinom(n, 1, p_y)

# Sanity check for binary values
if(any(y < 0 | y > 1 | !y %in% c(0, 1))) {
  cat("WARNING: Found non-binary values in y. Fixing...\n")
  y <- as.integer(y > 0.5)
}

# 3) Create Missing At Random (MAR) pattern for x1 based on x2, x3, y
# Missing probability depends on observed variables (MAR mechanism)
lp_miss <- 0.2 * x2 + 0.2 * x3 + 0.2 * y

# Scale to prevent extreme probabilities
lp_miss_scaled <- lp_miss / (2 * sd(lp_miss))

# Set intercept to achieve approximately 20% missingness
target_miss_rate <- 0.20
intercept <- qlogis(target_miss_rate) - mean(lp_miss_scaled)
p_miss <- plogis(intercept + lp_miss_scaled)

# Constrain probabilities to reasonable range [5%, 35%]
p_miss <- pmin(pmax(p_miss, 0.05), 0.35)

# Generate missing indicators and apply to x1
set.seed(456)  # Different seed for missing pattern
is_miss <- rbinom(n, 1, p_miss) == 1
x1 <- x1_true
x1[is_miss] <- NA

cat("Actual missingness rate in x1:", round(mean(is.na(x1)), 3), "\n")

# 4) Create data frames with proper structure

# Define data split sizes
central_size <- floor(n / 6)      # 5,000 observations
remote1_size <- floor(n / 3)      # 10,000 observations  
remote2_size <- n - central_size - remote1_size  # 15,000 observations

cat("Data split sizes:\n")
cat("  Central site:", central_size, "observations\n")
cat("  Remote site 1:", remote1_size, "observations\n") 
cat("  Remote site 2:", remote2_size, "observations\n")

# Check missingness distribution across sites before splitting
remote1_end <- central_size + remote1_size
central_miss_rate <- mean(is.na(x1[1:central_size]))
remote1_miss_rate <- mean(is.na(x1[(central_size + 1):remote1_end]))
remote2_miss_rate <- mean(is.na(x1[(remote1_end + 1):n]))

cat("Missing rates by site:\n")
cat("  Central:", round(100 * central_miss_rate, 1), "%\n")
cat("  Remote1:", round(100 * remote1_miss_rate, 1), "%\n") 
cat("  Remote2:", round(100 * remote2_miss_rate, 1), "%\n")
# Create main dataset with missing values
dat <- data.frame(
    x1 = x1,           # Binary with ~20% missing (MAR)
    x2 = x2,           # Continuous, complete
    x3 = x3,           # Continuous, complete  
    x4 = x4,           # Continuous, complete
    x5 = x5,           # Continuous, complete
    x6 = x6,           # Continuous, complete
    x7 = x7,           # Continuous with ~15% missing (random)
    x8 = x8,           # Continuous, complete
    x9 = x9,           # Continuous, complete  
    y = y              # Binary outcome, complete
)

# Create ground truth dataset (no missing values)
dat_truth <- data.frame(
    x1 = x1_true,      # True x1 values (no missing)
    x2 = x2,           # Same as observed
    x3 = x3,           # Same as observed
    x4 = x4,           # Same as observed 
    x5 = x5,           # Same as observed
    x6 = x6,           # Same as observed
    x7 = x7_linear,    # True x7 values (no missing)
    x8 = x8,           # Same as observed
    x9 = x9,           # Same as observed
    y = y              # Same as observed
)

# 5) Summary statistics and validation
cat("\n=== Data Generation Summary ===\n")
cat("Total observations:", n, "\n")
cat("Missing in x1:", sum(is.na(dat$x1)), "(", round(100*mean(is.na(dat$x1)), 1), "%)\n")
cat("Missing in x7:", sum(is.na(dat$x7)), "(", round(100*mean(is.na(dat$x7)), 1), "%)\n")

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

cat("\n=== Missing Value Distribution (x1) ===\n")
cat("Central site:", central_x1_miss, "missing (", 
    round(100 * central_x1_miss / nrow(central_data), 1), "% of central,",
    round(100 * central_x1_miss / total_x1_miss, 1), "% of all missing)\n")
cat("Remote site 1:", remote1_x1_miss, "missing (",
    round(100 * remote1_x1_miss / nrow(remote1_data), 1), "% of remote1,",
    round(100 * remote1_x1_miss / total_x1_miss, 1), "% of all missing)\n")
cat("Remote site 2:", remote2_x1_miss, "missing (",
    round(100 * remote2_x1_miss / nrow(remote2_data), 1), "% of remote2,",
    round(100 * remote2_x1_miss / total_x1_miss, 1), "% of all missing)\n")

cat("\nData generation completed successfully!\n")
