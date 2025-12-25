# Multiple Linear Regression with Backward Elimination (R)

# Import dataset
dataset = read.csv('50_Startups.csv')

# Encode categorical variable
dataset$State = factor(
  dataset$State,
  levels = c('New York', 'California', 'Florida'),
  labels = c(1, 2, 3)
)

# Split dataset
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Train model (all variables)
regressor = lm(
  formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
  # formula = Profit ~ ., 
  # "." means all independent variables.
  data = training_set
)

# Predict
y_pred = predict(regressor, newdata = test_set)

# Backward Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in 1:numVars) {
    regressor = lm(Profit ~ ., data = x)
    p_values = coef(summary(regressor))[-1, "Pr(>|t|)"]
    max_p = max(p_values)
    if (max_p > sl) {
      remove_var = names(which.max(p_values))
      x = x[, !(names(x) %in% remove_var)]
    }
  }
  return(summary(regressor))
}

SL = 0.05
backwardElimination(training_set, SL)

