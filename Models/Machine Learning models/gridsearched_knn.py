from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid_knn = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance']}

# Create a KNN classifier
knn_model = KNeighborsClassifier()

# Create a GridSearchCV object
grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn, cv=5, scoring='accuracy')

# Start the timer
start_timer()

# Perform grid search to find the best parameters and train the model
grid_search_knn.fit(X_train_resampled, y_train_resampled)

# Stop the timer and save the training time
training_time = stop_timer("KNN_Gridsearched")

# Get the best estimator from grid search
best_knn_model = grid_search_knn.best_estimator_

# Predict on training, validation, and testing sets
y_train_pred = best_knn_model.predict(X_train_scaled)
y_val_pred = best_knn_model.predict(X_val_scaled)
y_test_pred = best_knn_model.predict(X_test_scaled)

# Calculate accuracy for training, validation, and testing sets using original training data
training_accuracy = accuracy_score(y_train, y_train_pred)
validation_accuracy = accuracy_score(y_val, y_val_pred)
testing_accuracy = accuracy_score(y_test, y_test_pred)

# Generate classification reports for training, validation, and testing sets
training_classification_rep = classification_report(y_train, y_train_pred)
validation_classification_rep = classification_report(y_val, y_val_pred)
testing_classification_rep = classification_report(y_test, y_test_pred)

# Save the metrics using the provided function
save_accuracies("KNN_Gridsearched", training_accuracy, validation_accuracy, testing_accuracy, training_classification_rep, validation_classification_rep, testing_classification_rep)

# Print the metrics
print("Training Accuracy:", training_accuracy)
print("Training Classification Report:\n", training_classification_rep)
print("\nValidation Accuracy:", validation_accuracy)
print("Validation Classification Report:\n", validation_classification_rep)
print("\nTesting Accuracy:", testing_accuracy)
print("Testing Classification Report:\n", testing_classification_rep)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confuion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_val = confusion_matrix(y_val, y_val_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
