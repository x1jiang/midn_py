source("AVGMMICE/AVGMMICERemote.R")

# Remote site 1
conf = read.table("AVGMMICE/AVGMMICEConfRemote1",as.is=TRUE)
cent_host = conf[,1]
cent_port = conf[,2]
port = conf[,3]



X = matrix(rnorm(100000),10000,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
AVGMMICERemote(X,port,cent_host,cent_port)



source("AVGMMICE/AVGMMICERemote.R")

# Remote site 2
conf = read.table("AVGMMICE/AVGMMICEConfRemote2",as.is=TRUE)
cent_host = conf[,1]
cent_port = conf[,2]
port = conf[,3]



X = matrix(rnorm(100000),10000,10)
X[1:10,1] = NA
X[,2] = rbinom(100,1,0.5)
X[11:20,2] = NA
AVGMMICERemote(X,port,cent_host,cent_port)

