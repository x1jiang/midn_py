source("CSLMICE/CSLMICECentral.R")

conf = read.table("CSLMICE/CSLMICEConfCentral",as.is=TRUE)
hosts = conf[,1]
ports = conf[,2]
cent_ports = conf[,3]


X = matrix(rnorm(100000),10000,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
#imp = CSLMICECentral(X,10,1:2,c("Gaussian","Gaussian"),10,20,hosts,ports,cent_ports)
imp = CSLMICECentral(X,10,1:2,c("Gaussian","logistic"),10,20,hosts,ports,cent_ports)





