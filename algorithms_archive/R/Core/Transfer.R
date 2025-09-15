
readMat <- function(con)
{
  a = readBin(con,integer())
  b = readBin(con,integer())
  m = matrix(0,a,b)
  for ( i in 1:a )
    for ( j in 1:b )
      m[i,j] = readBin(con,numeric())
  m
}


writeMat <- function(m,con)
{
  a = dim(m)[1]
  b = dim(m)[2]
  writeBin(as.integer(a),con)
  writeBin(as.integer(b),con)
  for ( i in 1:a )
    for ( j in 1:b )
      writeBin(as.numeric(m[i,j]),con)
}


readVec <- function(con)
{
  a = readBin(con,integer())
  v = rep(0,a)
  for ( i in 1:a )
    v[i] = readBin(con,numeric())
  v
}


writeVec <- function(v,con)
{
  a = length(v)
  writeBin(as.integer(a),con)
  for ( i in 1:a )
    writeBin(as.numeric(v[i]),con)
}
