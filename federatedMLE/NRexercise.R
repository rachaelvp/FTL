# Federated Newton Raphson Exercise

################################ Centralized ###################################
# https://tomroth.com.au/logistic/

manual_logistic_regression = function(X, y, threshold = 1e-10, max_iter = 100)
  #A function to find logistic regression coeffiecients 
  #Takes three inputs: 
{
  #A function to return p, given X and beta
  #We'll need this function in the iterative section
  calc_p = function(X,beta)
  {
    beta = as.vector(beta)
    return(exp(X%*%beta) / (1+ exp(X%*%beta)))
  }  
  
  get_hessian = function(X, p)
  {
    #calculate matrix of weights W
    # W is n×n diagonal matrix where the i-th diagonal element of W is pi(1−pi)
    W = diag(p*(1-p))
    t(X)%*%W%*%X
  }
  
  get_score = function(X, y, p)
  {
    t(X)%*%(y - p)
  }
  #### setup bit ####
  
  #initial guess for beta
  beta = rep(0,ncol(X))
  
  #initial value bigger than threshold so that we can enter our while loop 
  diff = 10000 
  
  #counter to ensure we're not stuck in an infinite loop
  iter_count = 0
  
  #### iterative bit ####
  while(diff > threshold ) #tests for convergence
  {
    #calculate probabilities for logistic regression model using current beta estimate
    p = as.vector(calc_p(X,beta))
    
    #calculate first derivative (gradient) and second derivative of log-likelihood
    hessian = get_hessian(X,p)
    score = get_score(X,y,p)
    
    #calculate the change in beta
    # solve() function takes inverse of a matrix
    beta_change = solve(hessian) %*% score
    
    #update beta
    beta = beta + beta_change
    
    #calculate how much we changed beta by in this iteration 
    #if this is less than threshold, we'll break the while loop 
    diff = sum(beta_change^2)
    
    #see if we've hit the maximum number of iterations
    iter_count = iter_count + 1
    if(iter_count > max_iter) {
      stop("This isn't converging, mate.")
    }
  }
  #make it pretty 
  coef = c("(Intercept)" = beta[1], x1 = beta[2], x2 = beta[3], x3 = beta[4])
  print(paste0("Number of Newton-Raphson iterations: ", iter_count))
  return(coef)
}

################################## Federated ###################################

federated_logistic_regression = function(X1,y1,X2,y2,X3,y3,threshold = 1e-10, max_iter = 100){
  ###### utility functions used in iteration 
  calc_p = function(X,beta)
  {
    beta = as.vector(beta)
    return(exp(X%*%beta) / (1+ exp(X%*%beta)))
  }  
  get_hessian = function(X, p)
  {
    #calculate matrix of weights W
    # W is n×n diagonal matrix where the i-th diagonal element of W is pi(1−pi)
    W = diag(p*(1-p))
    t(X)%*%W%*%X
  }
  
  get_score = function(X, y, p)
  {
    t(X)%*%(y - p)
  }
  
  #### setup bit ####
  #initial guess for beta
  beta = rep(0,ncol(X))
  #initial value bigger than threshold so that we can enter our while loop 
  diff = 10000 
  #counter to ensure we're not stuck in an infinite loop
  iter_count = 0
  
  #### iterative bit ####
  while(diff > threshold ) #tests for convergence
  {
    #calculate probabilities using current estimate of beta
    p1 = as.vector(calc_p(X1,beta))
    p2 = as.vector(calc_p(X2,beta))
    p3 = as.vector(calc_p(X3,beta))
    
    #calculate first derivative (gradient) and second derivative of log-likelihood
    hessian1 = get_hessian(X1,p1)
    hessian2 = get_hessian(X2,p2)
    hessian3 = get_hessian(X3,p3)
    
    score1 = get_score(X1,y1,p1)
    score2 = get_score(X2,y2,p2)
    score3 = get_score(X3,y3,p3)
    
    #calculate the change in beta
    # solve() function takes inverse of a matrix
    beta_change = solve(hessian1+hessian2+hessian3) %*% (score1+score2+score3)
    
    #update beta
    beta = beta + beta_change
    
    #calculate how much we changed beta by in this iteration 
    #if this is less than threshold, we'll break the while loop 
    diff = sum(beta_change^2)
    
    #see if we've hit the maximum number of iterations
    iter_count = iter_count + 1
    if(iter_count > max_iter) {
      stop("This isn't converging, mate.")
    }
  }
  #make it pretty 
  coef = c("(Intercept)" = beta[1], x1 = beta[2], x2 = beta[3], x3 = beta[4])
  print(paste0("Number of Newton-Raphson iterations: ", iter_count))
  return(coef)
}

###################################### Test ####################################
set.seed(2016)
#simulate data 
#independent variables
x1 = rnorm(30,3,2) + 0.1*c(1:30)
x2 = rbinom(30, 1,0.3)
x3 = rpois(n = 30, lambda = 4)
x3[16:30] = x3[16:30] - rpois(n = 15, lambda = 2)

#dependent variable 
y = c(rbinom(5, 1,0.1),rbinom(10, 1,0.25),rbinom(10, 1,0.75),rbinom(5, 1,0.9))
x0 = rep(1,30) #bias
X = cbind(x0,x1,x2,x3)

# centralized logistic regression (using R package)
M1 = glm(y~x1+x2+x3, family = "binomial")
M1$coefficients

# centralized logistic regression (ours by hand)
manual_logistic_regression(X,y)

# our federated logistic regression
# make it federated first
X1 <- X[1:10,]
X2 <- X[11:20,]
X3 <- X[21:30,]
y1 <- y[1:10]
y2 <- y[11:20]
y3 <- y[21:30]
federated_logistic_regression(X1,y1,X2,y2,X3,y3)
