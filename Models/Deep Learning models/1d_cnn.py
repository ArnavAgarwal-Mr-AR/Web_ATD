from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Reshape data for 1D CNN
X_train_cnn = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Create a 1D CNN model
cnn_model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=X_train_cnn.shape[1:]),
    MaxPooling1D(2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_timer()
# Train the model
cnn_model.fit(X_train_cnn, y_train_resampled, epochs=40, batch_size=32, verbose=1)

training_time = stop_timer("1D-CNN")
# Predict probabilities
y_train_pred_prob_cnn = cnn_model.predict(X_train_cnn)
y_val_pred_prob_cnn = cnn_model.predict(X_val_cnn)
y_test_pred_prob_cnn = cnn_model.predict(X_test_cnn)

# Convert probabilities to binary predictions using a threshold
threshold = 0.5
y_train_pred_cnn = (y_train_pred_prob_cnn > threshold).astype(int)
y_val_pred_cnn = (y_val_pred_prob_cnn > threshold).astype(int)
y_test_pred_cnn = (y_test_pred_prob_cnn > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_cnn = accuracy_score(y_train_resampled, y_train_pred_cnn)
validation_accuracy_cnn = accuracy_score(y_val, y_val_pred_cnn)
testing_accuracy_cnn = accuracy_score(y_test, y_test_pred_cnn)

training_classification_rep_cnn = classification_report(y_train_resampled, y_train_pred_cnn)
validation_classification_rep_cnn = classification_report(y_val, y_val_pred_cnn)
testing_classification_rep_cnn = classification_report(y_test, y_test_pred_cnn)

# Save the metrics using the provided function
save_accuracies("1D_CNN", training_accuracy_cnn, validation_accuracy_cnn, testing_accuracy_cnn,
                training_classification_rep_cnn, validation_classification_rep_cnn, testing_classification_rep_cnn)

# Print the metrics
print("1D CNN - Training Accuracy:", training_accuracy_cnn)
print("1D CNN - Training Classification Report:\n", training_classification_rep_cnn)
print("\n1D CNN - Validation Accuracy:", validation_accuracy_cnn)
print("1D CNN - Validation Classification Report:\n", validation_classification_rep_cnn)
print("\n1D CNN - Testing Accuracy:", testing_accuracy_cnn)
print("1D CNN - Testing Classification Report:\n", testing_classification_rep_cnn)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#plotting the confuion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_cnn = confusion_matrix(y_train_resampled, y_train_pred_cnn)
confusion_matrix_val_cnn = confusion_matrix(y_val, y_val_pred_cnn)
confusion_matrix_test_cnn = confusion_matrix(y_test, y_test_pred_cnn)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_cnn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_cnn, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_cnn, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
