source("AVGMMI/AVGMMICentral.R")

conf = read.table("AVGMMI/AVGMMIConfCentral",as.is=TRUE)
hosts = conf[,1]
ports = conf[,2]
cent_ports = conf[,3]

# Continuous variable
X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA
imp = AVGMMICentral(X,10,1,"Gaussian",hosts,ports,cent_ports)

# Binary variable
X = matrix(rnorm(100000),10000,10)
X[,1] = rbinom(100,1,0.5)
X[1:10,1] = NA
imp = AVGMMICentral(X,10,1,"logistic",hosts,ports,cent_ports)


