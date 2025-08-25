source("AVGMMI/AVGMMIRemote.R")

# Remote site 1
conf = read.table("AVGMMI/AVGMMIConfRemote1",as.is=TRUE)
cent_host = conf[,1]
cent_port = conf[,2]
port = conf[,3]


# Generate data - continuous missing variable
X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA

# Generate data - binary missing variable
X = matrix(rnorm(1000),100,10)
X[,1] = rbinom(100,1,0.5)
X[1:10,1] = NA

# Start remote server
AVGMMIRemote(X,1,port,cent_host,cent_port)



# Remote site 2
conf = read.table("AVGMMI/AVGMMIConfRemote2",as.is=TRUE)
cent_host = conf[,1]
cent_port = conf[,2]
port = conf[,3]

# Generate data - continuous missing variable
X = matrix(rnorm(1000),100,10)
X[1:10,1] = NA

# Generate data - binary missing variable
X = matrix(rnorm(1000),100,10)
X[,1] = rbinom(100,1,0.5)
X[1:10,1] = NA

# Start remote server
AVGMMIRemote(X,1,port,cent_host,cent_port)

