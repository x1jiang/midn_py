source("IMICE/IMICECentral.R")

X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
imp = IMICECentral(X,10,1:2,c("Gaussian","logistic"),10,20)

