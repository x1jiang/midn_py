# Generate enhanced continuous test data for SIMI with 9 predictors

n <- 3000  # sample size
set.seed(42)  # For reproducibility

# 1) Generate core predictors with correlation structure

# Generate x2-x5 as independent variables first
x2 <- rnorm(n)  # standard normal
x3 <- rnorm(n)  # standard normal
x4 <- rnorm(n)  # standard normal
x5 <- rnorm(n)  # standard normal

# Generate x1 with linear relationship to x2-x5 (with small Gaussian noise)
# This will be the variable that has missing values
x1_true <- 0.22 * x2 + 0.12 * x3 + 0.32 * x4 + 0.22 * x5 + rnorm(n, mean = 0, sd = 0.5)

# Generate x6-x9 as independent variables
x6 <- rnorm(n)  # standard normal
x7 <- rnorm(n)  # standard normal
x8 <- rnorm(n)  # standard normal
x9 <- rnorm(n)  # standard normal

# 2) Generate continuous outcome y based on all predictors
# We'll use a combination of all variables with some being more important
y_true <- 1.0 + 0.8*x1_true + 0.6*x2 + 0.5*x3 + 0.4*x4 + 0.3*x5 + 
          0.2*x6 + 0.2*x7 + 0.1*x8 + 0.1*x9
          
# Add 20% error/noise to the outcome
noise <- rnorm(n, mean=0, sd=0.2*sd(y_true))
y <- y_true + noise

# 3) Create missingness in x1 (≈20% MAR based on x2, x3, y)
lp <- 0.4 * x2 + 0.3 * x3 + 0.3 * y
int <- qlogis(0.20) - mean(lp)  # Target 20% missingness
p_miss <- plogis(int + lp)

# Impose missingness
x1 <- x1_true
is_miss <- rbinom(n, 1, p_miss) == 1
x1[is_miss] <- NA            

# 4) Put it all in data frames
dat <- data.frame(
    x1 = x1,
    x2 = x2, 
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    x7 = x7,
    x8 = x8,
    x9 = x9,
    y = y
)

# Create ground truth dataframe with complete data
dat_full <- data.frame(
    x1 = x1_true,
    x2 = x2,
    x3 = x3,
    x4 = x4,
    x5 = x5,
    x6 = x6,
    x7 = x7,
    x8 = x8,
    x9 = x9,
    y = y
)

# Quick check: proportion missing in x1
cat("Proportion missing in x1:", mean(is.na(dat$x1)), "\n")

# Check correlation matrix of the full data
cat("\nCorrelation matrix of variables (without missing values):\n")
print(round(cor(dat_full[, 1:9]), 2))

# View first few rows
print(head(dat))

# Split the data into three parts
# 1-500: Central site data
# 501-1500: Remote site 1 data
# 1501-3000: Remote site 2 data

# Central site data (rows 1-500)
central_data <- dat[1:500, ]
central_data_truth <- dat_full[1:500, ]
write.csv(central_data, file = "./continuous_central.csv", row.names = FALSE)
write.csv(central_data_truth, file = "./continuous_central_truth.csv", row.names = FALSE)

# Remote site 1 data (rows 501-1500)
remote1_data <- dat[501:1500, ]
remote1_data_truth <- dat_full[501:1500, ]
write.csv(remote1_data, file = "./continuous_remote1.csv", row.names = FALSE)
write.csv(remote1_data_truth, file = "./continuous_remote1_truth.csv", row.names = FALSE)

# Remote site 2 data (rows 1501-3000)
remote2_data <- dat[1501:3000, ]
remote2_data_truth <- dat_full[1501:3000, ]
write.csv(remote2_data, file = "./continuous_remote2.csv", row.names = FALSE)
write.csv(remote2_data_truth, file = "./continuous_remote2_truth.csv", row.names = FALSE)

# Print summary of data split
cat("Data has been split into three files:\n")
cat("Central site data (rows 1-500):", nrow(central_data), "observations\n")
cat("Remote site 1 data (rows 501-1500):", nrow(remote1_data), "observations\n")
cat("Remote site 2 data (rows 1501-3000):", nrow(remote2_data), "observations\n")

# Display summary statistics
cat("\nSummary of x1 (with missingness):\n")
print(summary(dat$x1))
cat("Missing values in x1:", sum(is.na(dat$x1)), "(", round(100*mean(is.na(dat$x1)), 1), "%)\n")

# Add some summary statistics to validate the data generation
cat("\nBasic model fit on complete data:\n")
model_summary <- summary(lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, data = dat_full))
print(model_summary$coefficients)

# Visualize correlation structure and missing data patterns
# Check if we can create plots
if (requireNamespace("ggplot2", quietly = TRUE) && requireNamespace("GGally", quietly = TRUE)) {
  library(ggplot2)
  library(GGally)
  
  # Create directory for visualizations if it doesn't exist
  dir.create("./visualizations", recursive = TRUE, showWarnings = FALSE)
  
  # Correlation plot for the first 6 variables (x1-x5 + y)
  png("./visualizations/continuous_correlations.png", width = 900, height = 900, res = 100)
  print(GGally::ggpairs(dat_full[, c(1:5, 10)], 
                        title = "Correlation Structure of Variables (Continuous Outcome)"))
  dev.off()
  cat("\nCreated correlation visualization in ./visualizations/\n")
  
  # Missingness pattern visualization
  png("./visualizations/continuous_missing_patterns.png", width = 800, height = 400, res = 100)
  par(mfrow = c(1, 2))
  hist(x1_true[is_miss], main = "Distribution of x1 when missing", 
       xlab = "x1 value", col = "lightblue", border = "white")
  hist(x1_true[!is_miss], main = "Distribution of x1 when observed", 
       xlab = "x1 value", col = "lightgreen", border = "white")
  dev.off()
  cat("Created missing data pattern visualization\n")
} else {
  cat("\nNote: Install ggplot2 and GGally packages for visualizations\n")
}

# Optional code to run mice imputation (commented out)
# Uncomment if you want to run imputation

# library(mice)
# 
# # Create predictor matrix
# pred <- make.predictorMatrix(dat)
# 
# # Set imputation method
# meth <- make.method(dat)
# meth["x1"] <- "norm"  # Use normal model for x1
# 
# # Run multiple imputation
# imp <- mice(dat, m = 20, maxit = 5, method = meth, predictorMatrix = pred)
# 
# # Fit model to each imputed dataset
# fit <- with(imp, lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, data = dat))
# 
# # Pool results
# results <- pool(fit)
# print(summary(results))

