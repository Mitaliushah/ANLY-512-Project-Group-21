###########################################################################################################
# ANLY 512 Poster: Predictive Models without RFM Label
# Project Team : 21
# Dataset: eCommerce behavior data
# Dataset link: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
###########################################################################################################



# Import libraries
library(h2o)
library(ggplot2)
library(purrr)
library(dplyr)
library(tidyverse)
library(caret)



################### READ DATASET ##########################################
ecommerce_data <- read.csv("final_data.csv", stringsAsFactors = TRUE)
str(ecommerce_data)
ecommerce_data$is_purchased <- as.factor(ecommerce_data$is_purchased)
ecommerce_data$X <- NULL



# shuffle the dataframe by rows
ecommerce_data <- ecommerce_data[sample(1:nrow(ecommerce_data)), ]


# Split the data frame into Train and Test dataset (75:25) split
smp_size <- floor(0.75 * nrow(ecommerce_data))


# set the seed to make your partition reproducible
set.seed(5)
train_ind <- sample(seq_len(nrow(ecommerce_data)), size = smp_size)
train_df <- ecommerce_data[train_ind, ]
test_df <- ecommerce_data[-train_ind, ]


# initialize the h2o
h2o.init()


# create the train and test h2o data frames
train_df_h2o <- as.h2o(train_df)
test_df_h2o <- as.h2o(test_df)


# Identify predictors and response
y <- "is_purchased"
x <- setdiff(names(train_df_h2o), y)


# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5


# 1. Generate a 3-model ensemble (GBM + RF + Logistic)
# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_df_h2o,
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = 5
)



# Train & Cross-validate a Random Forest
my_rf <- h2o.randomForest(
  x = x,
  y = y,
  training_frame = train_df_h2o,
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = 5
)



# Train & Cross-validate a LR
my_lr <- h2o.glm(
  x = x,
  y = y,
  training_frame = train_df_h2o,
  family = c("binomial"),
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = 5
)


# Train a stacked LR ensemble using the GBM, RF and LR above
ensemble <- h2o.stackedEnsemble(
  x = x,
  y = y,
  metalearner_algorithm = "glm",
  training_frame = train_df_h2o,
  base_models = list(my_gbm, my_rf, my_lr)
)



# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test_df_h2o)




# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test_df_h2o)
perf_rf_test <- h2o.performance(my_rf, newdata = test_df_h2o)
perf_lr_test <- h2o.performance(my_lr, newdata = test_df_h2o)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test), h2o.auc(perf_lr_test))


# calculate AUC values
# for stacked ensemble
(ensemble_auc_test <- h2o.auc(perf))
# for GBM
h2o.auc(perf_gbm_test)
# for RF
h2o.auc(perf_rf_test)
# for LR
h2o.auc(perf_lr_test)

print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))



############################# ROC CURVE ################################

plot(perf_gbm_test, main = "ROC curve", col = "blue")
plot(perf_rf_test, add = TRUE, col = "red", lwd = 2, type = "roc")
plot(perf_lr_test, add = TRUE, col = "orange", lwd = 1.5)
plot(perf, add = TRUE, col = "green", lwd = 1.5)
legend(0.35, 0.25, c(" GBM", "RF", "LR", "Ensemble"), bty = "n", lwd = 1.2, cex = 0.75, col = c("blue", "red", "orange", "green"))




############################ Plot ROC using ggplot #######################

# Theme definition for ggplot
th <- ggplot2::theme(
  plot.title = element_text(color = "#003366", size = 12, face = "bold"),
  axis.title.x = element_text(color = "black", size = 11, face = "bold"),
  axis.title.y = element_text(color = "black", size = 11, face = "bold"),
  panel.background = element_rect(
    fill = "white",
    colour = "white",
    size = 0.5, linetype = 2
  ),
  panel.grid.major = element_line(
    size = 0.5, linetype = 2,
    colour = "gray80"
  ),
  panel.grid.minor = element_line(
    size = 0.25, linetype = 2,
    colour = "gray80"
  ),
  axis.line = element_line(size = 0.53, colour = "black"),
  panel.border = element_rect(linetype = 1, fill = NA, size = 0.53),
  axis.text = element_text(colour = "black", face = "bold", size = 10)
)


# save the plot
tiff(filename = "ROC-withoutRFM.png", width = 3500, height = 2000, res = 600)

# create plot
list(my_gbm, my_rf, my_lr, ensemble) %>%
  map(function(x) {
    x %>%
      h2o.performance() %>%
      .@metrics %>%
      .$thresholds_and_metric_scores %>%
      .[c("tpr", "fpr")] %>%
      add_row(tpr = 0, fpr = 0, .before = T) %>%
      add_row(tpr = 0, fpr = 0, .before = F)
  }) %>%
  map2(
    c("Gradient Boosting, AUC= 0.861", "Random Forest, AUC=0.860", "Logistic Regression, AUC=0.770", "Stacked Ensemble, AUC=0.863"),
    function(x, y) x %>% add_column(model = y)
  ) %>%
  reduce(rbind) %>%
  ggplot(aes(fpr, tpr, col = model)) +
  geom_line(size = 1.1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), linetype = 3, col = "black") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  ggtitle("ROC Curve for Stacked Ensemble Models")

dev.off()




################ Predictions on the trained model ##############
# Generate predictions on the test data set

# Using stacked ensemble model
pred <- h2o.predict(ensemble, newdata = test_df_h2o)
prediction_ensemble <- as.data.frame(pred$predict)
str(prediction_ensemble)
actual <- as.data.frame(test_df_h2o$is_purchased)
stacked <- cbind(actual, prediction_ensemble)
# save the predictions as a dataframe
write.csv(stacked, "No-RFM//stacked_ensemble.csv", row.names = FALSE)


# Using base Gradient Boosting model
pred <- h2o.predict(my_gbm, newdata = test_df_h2o)
prediction_ensemble <- as.data.frame(pred$predict)
str(prediction_ensemble)
actual <- as.data.frame(test_df_h2o$is_purchased)
stacked <- cbind(actual, prediction_ensemble)
# save the predictions as a dataframe
write.csv(stacked, "No-RFM/gradient-boosting.csv", row.names = FALSE)


# Using base RF model
pred <- h2o.predict(my_rf, newdata = test_df_h2o)
prediction_ensemble <- as.data.frame(pred$predict)
str(prediction_ensemble)
actual <- as.data.frame(test_df_h2o$is_purchased)
stacked <- cbind(actual, prediction_ensemble)
# save the predictions as a dataframe
write.csv(stacked, "No-RFM/random-forest.csv", row.names = FALSE)


# Using base LR model
pred <- h2o.predict(my_lr, newdata = test_df_h2o)
prediction_ensemble <- as.data.frame(pred$predict)
str(prediction_ensemble)
actual <- as.data.frame(test_df_h2o$is_purchased)
stacked <- cbind(actual, prediction_ensemble)
# save the predictions as a dataframe
write.csv(stacked, "No-RFM/logisticRegression.csv", row.names = FALSE)
