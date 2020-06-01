## Detecting Credit Card Fraud


# Importing the dataset from the respective folder

creditcard_data <- read.csv("E:/Study time/Data Science/R/Credit-Card-Dataset/creditcard.csv")

# Glance at the structure of the dataset

str(creditcard_data)

# Convert Class to a factor variable

creditcard_data$Class <- as.factor(creditcard_data$Class)

# Get the summary of the data

summary(creditcard_data)

# Count the missing value

sum(is.na(creditcard_data))

#------------------------------------------------------------#

# Get the distribution of fraud and legit transactions in the dataset

table(creditcard_data$Class)

# Get the percentage of fraud and legit transactions in the dataset

prop.table(table(creditcard_data$Class))

# Pie chart of credit card transactions 

labels <- c("legit","fraud")
labels <- paste(labels, round(100*prop.table(table(creditcard_data$Class)),2))
labels <- paste0(labels,"%")

pie(prop.table(table(creditcard_data$Class)),labels, col = c("orange","red"),
    main = "Pie chart of credit card transactions")

#------------------------------------------------------------#

# No model predictions

predictions <- rep.int(0, nrow(creditcard_data))
predictions <- as.factor(predictions)

install.packages("caret")
library(caret)

confusionMatrix(data = predictions, reference = creditcard_data$Class)

#------------------------------------------------------------#

library(dplyr)

set.seed(1)
creditcard_data <- creditcard_data %>% sample_frac(0.1)

table(creditcard_data$Class)

library(ggplot2)

ggplot(data = creditcard_data, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2','red'))

#------------------------------------------------------------#

# Creating training and test sets for fraud detection model

install.packages("caTools")
library(caTools)

set.seed(123)
data_sample = sample.split(creditcard_data$Class, SplitRatio = 0.8)

train_data = subset(creditcard_data, data_sample == TRUE)

test_data = subset(creditcard_data, data_sample == FALSE)

dim(train_data)
dim(test_data)

#------------------------------------------------------------#

# Balancing unbalacing data

# Random Over-Sampling (ROS)

table(train_data$Class)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit # = 22750 / 0.5

install.packages("ROSE")
library(ROSE)

oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = 'over',
                                   N = new_n_total,
                                   seed = 2020)

oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# Random Under-Sampling (RUS)

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud # = 35 / 0.5

install.packages("ROSE")
library(ROSE)

undersampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = 'under',
                                   N = new_n_total,
                                   seed = 2020)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# Random Under-Sampling (RUS)

table(train_data$Class)

n_new <- nrow(train_data) # =22785
fraction_fraud_new <- 0.50


install.packages("ROSE")
library(ROSE)

sampling_result <- ovun.sample(Class ~ .,
                                 data = train_data,
                                 method = 'both',
                                 N = n_new,
                                 p = fraction_fraud_new,
                                 seed = 2020)

sampled_credit <- sampling_result$data

table(sampled_credit$Class)

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# Using SMOTE to Balance the dataset (no duplicated)

install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)

# Set the number of fraud and legitimate cases, and the desired percentage of legitimate cases

n0 <- 22750
n1 <- 35
r0 <- 0.6

# Calculate the value for the dup_size parameter of SMOTE

ntimes <- ((1 - r0)/r0) * (n0/n1) - 1

smote_output <- SMOTE(X = train_data[, -c(1,31)],
                      target = train_data$Class,
                      K = 5,
                      dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

# Class distribution for original dataset

ggplot(data = train_data, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

# Class distribution for over-sampling dataset using SMOTE

ggplot(data = credit_smote, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

#------------------------------------------------------------#

# Decision Tree using SMOTE

install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ .,credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes

predicted_val <- predict(CART_model, test_data[-1], type = 'class')

# Build confusion matrix (caret)

library(caret)
confusionMatrix(predicted_val, test_data$Class)

predicted_val <- predict(CART_model, creditcard_data[-1], type = 'class')
confusionMatrix(predicted_val, creditcard_data$Class)

#------------------------------------------------------------#

# Decision Tree using SMOTE (not balancing the dataset)

CART_model <- rpart(Class ~ .,train_data[,-1])

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

# Predict fraud classes

predicted_val <- predict(CART_model, test_data[-1], type = 'class')

# Build confusion matrix (caret)

library(caret)
confusionMatrix(predicted_val, test_data$Class)

predicted_val <- predict(CART_model, creditcard_data[-1], type = 'class')
confusionMatrix(predicted_val, creditcard_data$Class)