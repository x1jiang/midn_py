source("IMI/IMICentral.R")

# Continuous variable
X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA
imp = IMICentral(X,10,1,"Gaussian")

# Binary variable
X = matrix(rnorm(100000),10000,10)
X[,1] = rbinom(100,1,0.5)
X[1:10,1] = NA
imp = IMICentral(X,10,1,"logistic")


