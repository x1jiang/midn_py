source("SIMICE/SIMICECentral.R")

conf = read.table("SIMICE/SIMICEConfCentral",as.is=TRUE)
hosts = conf[,1]
ports = conf[,2]
cent_ports = conf[,3]


X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
imp = SIMICECentral(X,10,1:2,c("Gaussian","logistic"),10,20,hosts,ports,cent_ports)




