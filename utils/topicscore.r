norm_score <- function(K, K0, m, D, Mquantile=0, scatterplot=FALSE, VHMethod = 'SVS'){
  library('rARPACK')
  library('nnls')
  p <- dim(D)[1]
  n <- dim(D)[2]
  M <- rowMeans(D)
  M_trunk <- pmin(M,quantile(M,Mquantile))
  
  obj <- svds(sqrt(M_trunk^(-1))*D, K)
  Xi <- obj$u
  
  #Step 1
  Xi[,1] <- abs(Xi[,1])
  R <- apply(Xi[, 2:K, drop = FALSE],2,function(x) x/Xi[,1])
  
  #Step 2
  vertices_est_obj <- vertices_est(R,K0,m)
  V <- vertices_est_obj$V
  theta <- vertices_est_obj$theta
  
  #Step 3
  Pi <- cbind(R, rep(1,p))%*%solve(cbind(V,rep(1,K)))
  Pi <- pmax(Pi,0)
  temp <- rowSums(Pi)
  Pi <- apply(Pi,2,function(x) x/temp)
  
  #Step 4
  A_hat <- sqrt(M_trunk)*Xi[,1]*Pi
  
  #Step 5
  temp <- colSums(A_hat)
  A_hat <- t(apply(A_hat,1,function(x) x/temp))
  
  return(A_hat)
}


vertices_est <- function(R,K0,m){
  library(quadprog)
  K <- dim(R)[2] + 1
  
  #Step 2a
  obj <- kmeans(R,m,iter.max=K*100,nstart = K*10)
  
  theta <- as.matrix(obj$centers)
  theta_original <- theta
  #plot(R[,1],R[,2])
  #points(theta[,1], theta[,2], col=2,lwd=4)
  
  #Step 2b'
  inner <- theta%*%t(theta)
  distance <- diag(inner)%*%t(rep(1,length(diag(inner)))) + rep(1,length(diag(inner)))%*%t(diag(inner)) - 2*inner
  top2 <- which(distance==max(distance),arr.ind=TRUE)[1,]
  theta0 <- as.matrix(theta[top2,])
  theta <- as.matrix(theta[-top2,])
  
  if (K0 > 2){
    for (k0 in 3:K0){
      inner <- theta%*%t(theta)
      distance <- rep(1,k0-1)%*%t(diag(inner))-2*theta0%*%t(theta)
      ave_dist <- colMeans(distance)
      index <- which(ave_dist==max(ave_dist))[1]
      theta0 <- rbind(theta0, theta[index,])
      theta <- as.matrix(theta[-index,])
    }
    theta <- theta0
  }
  
  #Step 2b
  comb <- combn(1:K0, K)
  max_values <- rep(0, dim(comb)[2])
  for (i in 1:dim(comb)[2]){
    for (j in 1:K0){
      max_values[i] <- max(simplex_dist(as.matrix(theta[j,]), as.matrix(theta[comb[,i],])), max_values[i])
    }
  }
  
  min_index <- which(max_values == min(max_values))
  
  #plot(theta[,1],theta[,2])
  #points(theta[comb[,min_index],1],theta[comb[,min_index],2],col=2,pch=2)
  
  return(list(V=theta[comb[,min_index[1]],], theta=theta_original))
}

simplex_dist <- function(theta, V){
  #library(Matrix)
  VV <- cbind(diag(rep(1,dim(V)[1]-1)), -rep(1,dim(V)[1]-1))%*%V
  D <- VV%*%t(VV)
  d <- VV%*%(theta-V[dim(V)[1],])
  
  A <- cbind(diag(rep(1,dim(V)[1]-1)), -rep(1,dim(V)[1]-1))
  b0 <- c(rep(0,dim(V)[1]-1),-1)
  
  # D <- matrix(nearPD(D)$mat, nrow(D), ncol(D))
  # D <- nearPD(D)
  obj <- solve.QP(D, d, A, b0)
  return(sum((theta-V[dim(V)[1],]) ^2)+ 2*obj$value)
}