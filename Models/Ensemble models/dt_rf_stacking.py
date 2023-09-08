from mlxtend.classifier import StackingClassifier

# Create a stacking classifier using Decision Tree and Random Forest as base models
stacked_classifier = StackingClassifier(classifiers=[decision_tree_model, random_forest_model], meta_classifier=random_forest_model)

# Start the timer
start_timer()

# Train the stacking classifier on the validation data
stacked_classifier.fit(X_val_scaled, y_val)

training_time_stacked = stop_timer("ENS_STACKING_DT_RF_Normal")

y_train_pred_stacked = stacked_classifier.predict(X_train_scaled)

y_val_pred_stacked = stacked_classifier.predict(X_val_scaled)

y_test_pred_stacked = stacked_classifier.predict(X_test_scaled)

training_accuracy_stacked = accuracy_score(y_train, y_train_pred_stacked)
validation_accuracy_stacked = accuracy_score(y_val, y_val_pred_stacked)
testing_accuracy_stacked = accuracy_score(y_test, y_test_pred_stacked)

training_classification_rep_stacked = classification_report(y_train, y_train_pred_stacked)
validation_classification_rep_stacked = classification_report(y_val, y_val_pred_stacked)
testing_classification_rep_stacked = classification_report(y_test, y_test_pred_stacked)

save_accuracies(
    "ENS_STACKING_DT_RF_Normal",
    training_accuracy_stacked,
    validation_accuracy_stacked,
    testing_accuracy_stacked,
    training_classification_rep_stacked,
    validation_classification_rep_stacked,
    testing_classification_rep_stacked,
)

# Print the metrics for Stacking Classifier
print("\nStacking Classifier Training Accuracy:", training_accuracy_stacked)
print("Stacking Classifier Training Classification Report:\n", training_classification_rep_stacked)
print("\nStacking Classifier Validation Accuracy:", validation_accuracy_stacked)
print("Stacking Classifier Validation Classification Report:\n", validation_classification_rep_stacked)
print("\nStacking Classifier Testing Accuracy:", testing_accuracy_stacked)
print("Stacking Classifier Testing Classification Report:\n", testing_classification_rep_stacked)

print("Stacking Classifier Training Time:", training_time_stacked, "seconds")

#Plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Predict on training, validation, and testing sets
y_train_pred_stacked = stacked_classifier.predict(X_train_scaled)
y_val_pred_stacked = stacked_classifier.predict(X_val_scaled)
y_test_pred_stacked = stacked_classifier.predict(X_test_scaled)

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_stacked = confusion_matrix(y_train, y_train_pred_stacked)
confusion_matrix_val_stacked = confusion_matrix(y_val, y_val_pred_stacked)
confusion_matrix_test_stacked = confusion_matrix(y_test, y_test_pred_stacked)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_stacked, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix (Stacking Classifier)')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_stacked, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix (Stacking Classifier)')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_stacked, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Testing Set Confusion Matrix (Stacking Classifier)')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
