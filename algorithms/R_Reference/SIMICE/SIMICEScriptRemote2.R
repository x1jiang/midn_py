source("SIMICE/SIMICERemote.R")

# Remote site 2
conf = read.table("SIMICE/SIMICEConfRemote2",as.is=TRUE)
cent_host = conf[,1]
cent_port = conf[,2]
port = conf[,3]



X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
SIMICERemote(X,port,cent_host,cent_port)

