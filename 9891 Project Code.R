rm(list = ls())
cat("\014")

set.seed(125)
setwd("put-your-path-here")

library(doMC) # to install, use: install.packages("doMC", repos="http://R-Forge.R-project.org")
library(parallel) # for parallel computing
library(glmnet)
library(tidyverse)
library(randomForest)
library(gridExtra)
library(readr)
library(ggplot2)
library(pROC)

num_cores = "your number of cpu cores"
registerDoMC(cores = num_cores)
############################################################################
#                Read In & Prepare the Dataset for Analysis                #
############################################################################
data <- read_csv("caravan.csv") %>% as.data.frame()

# All vars categorical. Use model.matrix to expand and one-hot-encode the feature space

data.X <- data[,-86] 
data.X <- lapply(data.X, factor)
data.X <- model.matrix(~.-1, data = data.X) # using -1 means don't create a column of 1's for the intercept

data.Y <- as.factor(data$CARAVAN)

# put it all together for use with Random Forest. 
data.full <- as.data.frame(data.X)
data.full$CARAVAN <- data.Y

n = nrow(data)
p = dim(data.X)[2]

############################################################################
#                         Initialize Storage Matrices                      #
############################################################################

runs = 50  

## AUC Data frame

auc.initialize <- rep(0, runs)  # initialize a vector to be used for the columns
                                
Run   = seq(1, runs)
ElNet = auc.initialize
Lasso = auc.initialize
Ridge = auc.initialize
RF    = auc.initialize

auc.df <- data.frame(Run   = Run,
                     ElNet.train = ElNet,  # 2
                     Lasso.train = Lasso,  # 3
                     Ridge.train = Ridge,  # 4
                     RF.train    = RF,     # 5
                     ElNet.test  = ElNet,  # 6
                     Lasso.test  = Lasso,  # 7
                     Ridge.test  = Ridge,  # 8
                     RF.test     = RF)     # 9

## CV Times Data frame

times = data.frame(Run = Run, 
                   ElNet = rep(0, runs), 
                   Lasso = rep(0, runs), 
                   Ridge = rep(0, runs),
                   RF    = rep(0, runs))

## Lambdas Data Frame

lambdas.df <- data.frame(Run = runs, 
                         ElNet = rep(0,runs),
                         Lasso = rep(0,runs),
                         Ridge = rep(0,runs))

# Individual run times for each loop

run.times <- data.frame(Loop = seq(1,50),
                        Time = rep(0,50))


# alpha parameters
elnet.alpha = 0.5
lasso.alpha = 1
ridge.alpha = 0
############################################################################
#                             50-Run Simulation                            #
############################################################################
sim.start <- proc.time()
for(i in 1:runs){
  cat("Cross Validation Number",i,"\n")
  
  loop.start = proc.time()
  cat("Generating Sample", i, "of 50", "\n")
  
  ##### Sample & Create Train/Test Indices #####
  n_obs         <- seq(1:n)
  sample.size   <- floor(.90*n)
  train.indices <- sample(seq_len(n), size=sample.size)
  test.indices  <- !(n_obs%in%train.indices)
  
  ##### Partition Training Data #####
  writeLines("Partitioning Training Data")
  X.train = data.X[train.indices,]
  Y.train = data.Y[train.indices]
  
  writeLines("Partitioning Test Data")
  X.test = data.X[test.indices,]
  Y.test = data.Y[test.indices]
  
  ##### Cross Validation #####
  cat("Beginning Elastic Net Cross Validation Number",i, "of 50", "\n")
  
  # Elastic Net
  elnet.start <- proc.time()
  elnet.cv    <- cv.glmnet(X.train, Y.train,
                           parallel     = TRUE, 
                           family       = "binomial",
                           alpha        = elnet.alpha, 
                           type.measure = "auc")
  elnet.end   <- proc.time()
  elnet.time  <- elnet.end[3]-elnet.start[3]
  cat("It Took", elnet.time, "seconds to cross validate Elastic Net on pass", i, "\n")
  times$ElNet[i]  <- elnet.time
  
  # Lasso 
  cat("Beginning Lasso Cross Validation Number", i, "of 50.", "\n" )
  lasso.start <- proc.time()
  lasso.cv    <- cv.glmnet(X.train, Y.train, 
                           parallel     = TRUE,
                           family       = "binomial",
                           alpha        = lasso.alpha, 
                           type.measure = "auc")
  lasso.end   <- proc.time()
  lasso.time  <- lasso.end[3]-lasso.start[3]
  cat("It Took", lasso.time, "seconds to cross validate Lasso on pass", i, "\n")
  times$Lasso[i]  <- lasso.time
 
   # Ridge
  cat("Beginning Ridge Cross Validation Number", i, "of 50.", "\n" )
  ridge.start <- proc.time()
  ridge.cv    <- cv.glmnet(X.train, Y.train, 
                           parallel     = TRUE,
                           family       = "binomial",
                           alpha        = ridge.alpha,
                           type.measure = "auc")
  ridge.end   <- proc.time()
  ridge.time  <- ridge.end[3]-ridge.start[3]
  cat("It Took", ridge.time, "seconds to cross validate Ridge on pass", i, "\n")
  times$Ridge[i]  <- ridge.time
  
  
  if(i == 5){
    writeLines("Plotting Cross Validation Curves")
    par(mfrow = c(3,1))
    plot(elnet.cv)
    title(main="Elastic-Net Cross Validation Curve",line = 3)
    plot(lasso.cv)
    title(main="Lasso Cross Validation Curve",line = 3)
    plot(ridge.cv)
    title(main="Ridge Cross Validation Curve",line = 3)
  }
 
  
  lambdas.df$ElNet[i] <- elnet.cv$lambda.min
  lambdas.df$Lasso[i] <- lasso.cv$lambda.min
  lambdas.df$Ridge[i] <- ridge.cv$lambda.min
  
  
  # Pull out the parameters
  cat("Extracting the Elastic Net Beta Coefficients for pass", i, "\n")
  elnet.index   <- which.max(elnet.cv$cvm)
  elnet.beta    <- as.vector(elnet.cv$glmnet.fit$beta[, elnet.index])
  elnet.beta0   <- elnet.cv$glmnet.fit$a0[elnet.index]
  
  cat("Extracting the Lasso Beta Coefficients for pass", i, "\n")
  lasso.index   <- which.max(lasso.cv$cvm)
  lasso.beta    <- as.vector(lasso.cv$glmnet.fit$beta[, lasso.index])
  lasso.beta0   <- lasso.cv$glmnet.fit$a0[lasso.index]
  
  cat("Extracting the Ridge Beta Coefficients for pass", i, "\n")
  ridge.index   <- which.max(ridge.cv$cvm)
  ridge.beta    <- as.vector(ridge.cv$glmnet.fit$beta[, ridge.index])
  ridge.beta0   <- ridge.cv$glmnet.fit$a0[ridge.index]
  
  ##### Calculate AUC #####
  cat("Calculating AUC for run", i, "of 50", "\n")
  writeLines("Calculating Distances From the Hyper Plane")
  # Calculate Distances from the hyper-plane
  xtb.elnet.train     <- X.train%*%elnet.beta + elnet.beta0
  xtb.lasso.train     <- X.train%*%lasso.beta + lasso.beta0
  xtb.ridge.train     <- X.train%*%ridge.beta + ridge.beta0
  
  xtb.elnet.test      <- X.test%*%elnet.beta + elnet.beta0
  xtb.lasso.test      <- X.test%*%lasso.beta + lasso.beta0
  xtb.ridge.test      <- X.test%*%ridge.beta + ridge.beta0
  
  writeLines("Calculating The Probability Matrices")
  # Calculate Probability Matrices
  elnet.probs.train = exp(xtb.elnet.train)/(1+exp(xtb.elnet.train))
  elnet.probs.test = exp(xtb.elnet.test)/(1+exp(xtb.elnet.test))
  
  lasso.probs.train = exp(xtb.lasso.train)/(1+exp(xtb.lasso.train))
  lasso.probs.test = exp(xtb.lasso.test)/(1+exp(xtb.lasso.test))
  
  ridge.probs.train = exp(xtb.ridge.train)/(1+exp(xtb.ridge.train))
  ridge.probs.test = exp(xtb.ridge.test)/(1+exp(xtb.ridge.test))
  
  # Calculate AUC for El-net, Lasso, Elastic Net & Store them for use later
  writeLines("Calculating and Storing the AUC")
  elnet.train.auc <- auc(roc(as.factor(Y.train), c(elnet.probs.train)))
  lasso.train.auc <- auc(roc(as.factor(Y.train), c(lasso.probs.train)))
  ridge.train.auc <- auc(roc(as.factor(Y.train), c(ridge.probs.train)))
  
  elnet.test.auc <- auc(roc(as.factor(Y.test), c(elnet.probs.test)))
  lasso.test.auc <- auc(roc(as.factor(Y.test), c(lasso.probs.test)))
  ridge.test.auc <- auc(roc(as.factor(Y.test), c(ridge.probs.test)))
  
  auc.df$ElNet.train[i] <- elnet.train.auc
  auc.df$Lasso.train[i] <- lasso.train.auc
  auc.df$Ridge.train[i] <- ridge.train.auc
  
  auc.df$ElNet.test[i] <- elnet.test.auc
  auc.df$Lasso.test[i] <- lasso.test.auc
  auc.df$Ridge.test[i] <- ridge.test.auc
  ##### Random Forest #####
  cat("Fitting Random Forest number", i, "of 50.", "\n")
  # Random Forest
  rf.start     <- proc.time()
  rf.train     <- randomForest(CARAVAN~., data = data.full[train.indices,], mtry=sqrt(p))  # model
  rf.end       <- proc.time()
  rf.time      <- rf.end[3]-rf.start[3]
  times$RF[i]  <- rf.time
  
  cat("Random Forest - Calculating Train and Test AUC", "\n" )
  preds.test   <- predict(rf.train, newdata = X.test, type="vote") # test predictions 
  
  
  rf.train.auc <- auc(roc(data.full$CARAVAN[train.indices], rf.train$votes[,2]))   # auc train
  rf.test.auc  <- auc(roc(data.full$CARAVAN[test.indices],  preds.test[,2]))       # auc test
  
  # Store AUC in the data frame above
  auc.df$RF.train[i] <- rf.train.auc
  auc.df$RF.test[i]  <- rf.test.auc
  ##### End #####
  loop.end = proc.time()
  loop.time = loop.end[3]-loop.start[3]
  run.times[i,2] <- loop.time
  cat("Loop ", i, "of 50 took ", loop.time, " seconds to complete", "\n")
}
sim.end <- proc.time()
total.sim.time <- sim.end[3]-sim.start[3]
total.sim.time 
###############################################
###             AUC Box-Plots              ###
###############################################

sample.train <- rep("Train", 200)
sample.test  <- rep("Test", 200)
elnet.label  <- rep("Elastic Net", 50)
lasso.label  <- rep("Lasso", 50)
ridge.label  <- rep("Ridge", 50)
rf.label     <- rep("Random Forest", 50)

method.labels = c(elnet.label, lasso.label, ridge.label, rf.label)

train_auc = c(auc.df$ElNet.train, auc.df$Lasso.train, auc.df$Ridge.train, auc.df$RF.train)
test_auc  = c(auc.df$ElNet.test, auc.df$Lasso.test,auc.df$Ridge.test,auc.df$RF.test)
total_auc = c(train_auc, test_auc)

train.auc.df = data.frame(Sample = sample.train, 
                          Model = method.labels, 
                          AUC = train_auc)
test.auc.df = data.frame(Sample = sample.test, 
                         Model = method.labels, 
                         AUC = test_auc)
long.df.auc = rbind(train.auc.df, test.auc.df)

# Create the boxplot
long.df.auc %>% ggplot(aes(x=Model, y = AUC, color = Model)) + geom_boxplot() + facet_wrap(~Sample)


###############################################
### Fit the 4 models on the entire data set ###
###############################################

methods = c("Elastic Net", "Lasso", "Ridge", "Random Forest")

elnet.median.test.auc <- median(auc.df$ElNet.test)
lasso.median.test.auc <- median(auc.df$Lasso.test)
ridge.median.test.auc <- median(auc.df$Ridge.test)
rf.median.test.auc    <- median(auc.df$RF.test)

##### Storage for Median Test AUC and Full Run Times #####
median.auc <- c(elnet.median.test.auc, lasso.median.test.auc,
                ridge.median.test.auc, rf.median.test.auc)
full.times <- c(rep(0,4))


##### Elastic Net #####

elnet.start.full <- proc.time()
elnet.full.cv    <- cv.glmnet(data.X, data.Y, 
                              parallel     = TRUE,
                              family       = "binomial",
                              alpha        = elnet.alpha,
                              type.measure = "auc")

elnet.full.lambda <- elnet.full.cv$lambda.min
elnet.model <- glmnet(data.X, data.Y, 
                      family="binomial",
                      alpha = elnet.alpha)
elnet.end.full <- proc.time()
elnet.full.time <- elnet.end.full[3] - elnet.start.full[3]
full.times[1] <- elnet.full.time



##### Lasso #####

lasso.start.full  <- proc.time()
lasso.full.cv     <- cv.glmnet(data.X, data.Y, 
                               parallel     = TRUE,
                               family       = "binomial",
                               alpha        = lasso.alpha,
                               type.measure = "auc")
lasso.full.lambda <- lasso.full.cv$lambda.min
lasso.model       <- glmnet(data.X, data.Y, 
                            family  = "binomial",
                            alpha   = lasso.alpha)
lasso.end.full    <- proc.time()
lasso.full.time   <- lasso.end.full[3] - lasso.start.full[3]
full.times[2]     <- lasso.full.time

##### Ridge #####

ridge.start.full  <- proc.time()
ridge.full.cv     <- cv.glmnet(data.X, data.Y, 
                               parallel     = TRUE,
                               family       = "binomial",
                               alpha        = ridge.alpha,
                               type.measure = "auc")
ridge.full.lambda <- ridge.full.cv$lambda.min
ridge.model       <- glmnet(data.X, data.Y, 
                            family="binomial",
                            alpha = ridge.alpha)
ridge.end.full    <- proc.time()
ridge.full.time   <- ridge.end.full[3] - ridge.start.full[3]
full.times[3]     <- ridge.full.time

###### Random Forest ######
rf.start.full     <- proc.time()
rf.full.model     <- randomForest(CARAVAN~., data = data.full, mtry = sqrt(p))
rf.end.full       <- proc.time()
rf.time           <- rf.end.full[3]-rf.start.full[3]
full.times[4]     <- rf.time

##### Collect AUC and Run Times into a Data Frame #####

auc.times.df <- data.frame(Method = methods,
                           AUC = median.auc,
                           Time = full.times)
auc.times.df

#####  Bar plots of the standardized coefficients  #####

s = apply(data.X, 2, sd) # get the standard deviation of the variables

elnet.full.beta <- as.vector(elnet.model$beta[,which.max(elnet.full.cv$cvm)])
lasso.full.beta <- as.vector(lasso.model$beta[,which.max(lasso.full.cv$cvm)])
ridge.full.beta <- as.vector(ridge.model$beta[,which.max(ridge.full.cv$cvm)])

elnet.coefs       <- elnet.full.beta*s  # multiply coefficients by sd(Variable) to standardize
lasso.coefs       <- lasso.full.beta*s
ridge.coefs       <- ridge.full.beta*s

rf.importance     <- importance(rf.full.model)        # pull the importance of the variables
row.names(rf.importance) <- NULL                      # remove rownames from the variable importance

VarNames          <- colnames(data.X)
VarNumber         <- as.character(seq(1:ncol(data.X)))         # label the variable by number for ease of display


variable.importance   <- data.frame(Variable = VarNames, 
                                    Number   = VarNumber,
                                    ElNet    = elnet.coefs, 
                                    Lasso    = lasso.coefs, 
                                    Ridge    = ridge.coefs, 
                                    RF       = rf.importance)

# Order Coefficients By desc(ElNet)
variable.importance   <- variable.importance[order(variable.importance$ElNet, decreasing = TRUE),]
# force ggplot to respect the order the data is sorted in. 
variable.importance$Number <- factor(variable.importance$Number, levels=variable.importance$Number) 

variable.importance_ELN   <- variable.importance[order(abs(variable.importance$ElNet), decreasing = TRUE),]
variable.importance_ELN

variable.importance_L   <- variable.importance[order(abs(variable.importance$Lasso), decreasing = TRUE),]
variable.importance_L

variable.importance_R   <- variable.importance[order(abs(variable.importance$Ridge), decreasing = TRUE),]
variable.importance_R

variable.importance_RF   <- variable.importance[order(abs(variable.importance$MeanDecreaseGini), decreasing = TRUE),]
variable.importance_RF

# create the plots so we can feed them to grid arrange
elnetPlot = variable.importance %>% ggplot(aes(x = Number, y = ElNet)) +
  geom_bar(stat = "identity", fill="white", colour="#FF0000") +
  labs(title = "Standardized Elastic Net Coefficients", x = "Variable", y = "Coefficient") + theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())

lassoPlot = variable.importance %>% ggplot(aes(x = Number, y = Lasso))  +
  geom_bar(stat = "identity", fill="white", colour="#70AD47") +
  labs(title = "Standardized Lasso Coefficients", x = "Variable", y = "Coefficient") + theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())

ridgePlot = variable.importance %>% ggplot(aes(x = Number, y = Ridge)) +
  geom_bar(stat = "identity", fill="white", colour="#CC04C2") +
  labs(title = "Standardized Ridge Coefficients", x = "Variable", y = "Coefficient") + theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank())

rfPlot = variable.importance %>% ggplot(aes(x = Number, y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill="white", colour="#02C9CE") +
  labs(title = "Random Forrest Variable Importance", x = "Variable", y = "Importance") + theme(axis.title.x=element_blank(), axis.text.x=element_blank(),axis.ticks.x=element_blank() )

 

# arrange the plots in a single image
grid.arrange(elnetPlot, lassoPlot, ridgePlot, rfPlot, nrow=4)



