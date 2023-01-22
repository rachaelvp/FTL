# Federated Stochastic Gradient Descent Exercise

# https://www.ocf.berkeley.edu/~janastas/stochastic-gradient-descent-in-r.html
# https://www.analyticsvidhya.com/blog/2021/04/gradient-descent-in-linear-regression/
# https://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/24-coord-desc.pdf
# https://courses.cs.washington.edu/courses/cse446/17wi/slides/lasso.pdf

linear_regression_gradient <- function(theta, y, X){
  n <- length(y)
  error <- t(y) - theta%*%t(X)
  -(2/n)%*%(error)%*%X
}
MSE <- function(theta, y, X){
  n <- length(y)
  error <- t(y) - theta%*%t(X)
  (1/n)%*%t(error)%*%error
}

gradient_descent <- function(X, y, 
                             gradient_function = linear_regression_gradient, 
                             risk_function = MSE, 
                             initial_theta = rep(0, ncol(X)+1),
                             threshold = 1e-10, 
                             learning_rate = 0.1,  
                             max_iter = 1000){
  
  # add intercept column
  if(is.null(colnames(X))){
    colnames(X) <- paste0("X", seq(1:ncol(X)))
  }
  X <- as.matrix(data.frame("Intercept" = rep(1, length(y)), X))
  
  # initiate coefficients 
  theta <- t(as.matrix(initial_theta))
  
  loss <- c()
  converged <- TRUE
  for(i in 1:max_iter){
    gradient_new <- gradient_function(theta, y, X)
    loss <- append(loss, risk_function(theta, y, X))
    if(any(is.nan(gradient_new) | is.infinite(gradient_new))){
      converged <- FALSE
      warning(paste0("NaN/Inf gradient after ", i, " steps. Failed to converge"))
      break
    } else {
      gradient <- gradient_new
      if(sqrt(sum(gradient^2)) <= threshold){
        break
      }
      theta <- theta - learning_rate*gradient
    }
  }
  if(converged){
    print(paste0("Algorithm converged in ", i, " steps"))
  }
  coefs <- as.numeric(theta)
  names(coefs) <- c("(Intercept)", colnames(X)[-1])
  list(coefs = coefs, loss = loss)
}
gradient_descent <- function(y, X, threshold = 1e-10, learning_rate = 0.1,  
                             max_iter = 1000){
  # add intercept column
  X <- as.matrix(data.frame("Intercept" = rep(1,length(y)), X))
  # initiate coefficients 
  theta <- t(as.matrix(rep(0, ncol(X))))
  
  get_gradient <- function(theta, y, X){
    n <- length(y)
    error <- t(y) - theta%*%t(X)
    -(2/n)%*%(error)%*%X
  }
  
  for(i in 1:max_iter){
    gradient_new <- get_gradient(theta, y, X)
    if(any(is.nan(gradient_new) | is.infinite(gradient_new))){
      break
    } else {
      gradient <- gradient_new
      if(sqrt(sum(gradient^2)) <= threshold){
        break
      }
      theta <- theta - learning_rate*gradient
    }
  }
  print(paste0("Algorithm converged in ", i, " steps"))
  as.numeric(theta)
}
subgradient_descent <- function(X, y, threshold = 1e-10, learning_rate = 0.1,  
                                max_iter = 1000){
  # add intercept column
  X <- as.matrix(data.frame("Intercept" = rep(1,length(y)), X))
  # initiate coefficients 
  theta <- t(as.matrix(rep(0, ncol(X))))
  
  get_subgradient <- function(theta, y, X){
    n <- length(y)
    error <- t(y) - theta%*%t(X)
    -(2/n)%*%(error)%*%X
  }
  get_MSE <- function(theta, y, X){
    n <- length(y)
    error <- t(y) - theta%*%t(X)
    (1/n)%*%t(error)%*%error
  }
  
  MSE <- c()
  for(i in 1:max_iter){
    gradient_new <- get_gradient(theta, y, X)
    MSE <- append(MSE, get_MSE(theta, y, X))
    if(any(is.nan(gradient_new) | is.infinite(gradient_new))){
      break
    } else {
      gradient <- gradient_new
      if(sqrt(sum(gradient^2)) <= threshold){
        break
      }
      theta <- theta - learning_rate*gradient
    }
  }
  print(paste0("Algorithm converged in ", i, " steps"))
  list(coefs = as.numeric(theta), MSE = MSE)
}

federated_gradient_descent <- function(y1, X1, y2, X2, y3, X3, 
                                       threshold = 1e-10, 
                                       learning_rate = 0.1,  
                                       max_iter = 1000){
  # add intercept column
  X1 <- as.matrix(data.frame("Intercept" = rep(1,length(y1)), X1))
  X2 <- as.matrix(data.frame("Intercept" = rep(1,length(y2)), X2))
  X3 <- as.matrix(data.frame("Intercept" = rep(1,length(y3)), X3))
  
  # initiate coefficients 
  theta <- t(as.matrix(rep(0, ncol(X1))))
  
  get_gradient <- function(theta, y, X){
    n <- length(y)
    error <- t(y) - theta%*%t(X)
    -(2/n)%*%(error)%*%X
  }
  for(i in 1:max_iter){
    grad1_new <- get_gradient(theta, y1, X1)
    grad2_new <- get_gradient(theta, y2, X2)
    grad3_new <- get_gradient(theta, y3, X3)
    gradient_new <- grad1_new+grad2_new+grad3_new
    if(any(is.nan(gradient_new) | is.infinite(gradient_new))){
      break
    } else {
      gradient <- gradient_new
      if(sqrt(sum(gradient^2)) <= threshold){
        break
      }
      theta <- theta - learning_rate*gradient
    }
  }
  print(paste0("Algorithm converged in ", i, " steps"))
  as.numeric(theta)
}

set.seed(5532)
x1 = rnorm(n = 900)
x2 = rnorm(n = 900)
x3 = rnorm(n = 900)
x4 = rnorm(n = 900)
x5 = rnorm(n = 900)
y = x1 + x2^2 + 1/x3 + x1*x4 + rnorm(900)

GD_coefs <- gradient_descent(y = y, X = data.frame(x1,x2,x3,x4,x5), 
                             learning_rate = 0.5)
lm_coefs <- lm(y ~., data = data.frame(y, x1,x2,x3, x4,x5))$coefficients
all.equal(as.numeric(lm_coefs), GD_coefs)

X <- data.frame(x1,x2,x3,x4,x5)
FGD_coefs <- federated_gradient_descent(
  y1 = y[1:300], X1 =X[1:300,], y2 = y[301:600], X2 =X[301:600,], 
  y3 = y[601:900], X3 =X[601:900,], learning_rate = 0.15
)
all.equal(as.numeric(lm_coefs), FGD_coefs)
