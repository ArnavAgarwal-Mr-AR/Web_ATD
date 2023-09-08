from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Create a GRU model
gru_model = Sequential([
    GRU(64, activation='relu', input_shape=X_train_rnn.shape[1:], return_sequences=True),
    Dropout(0.5),
    GRU(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Start timer
start_timer()

# Train the model
gru_model.fit(X_train_rnn, y_train_resampled, epochs=40, batch_size=32, verbose=1)

# Stop timer
training_time = stop_timer("GRU")

# Predict probabilities
y_train_pred_prob_gru = gru_model.predict(X_train_rnn)
y_val_pred_prob_gru = gru_model.predict(X_val_rnn)
y_test_pred_prob_gru = gru_model.predict(X_test_rnn)

# Convert probabilities to binary predictions using a threshold
threshold = 0.5
y_train_pred_gru = (y_train_pred_prob_gru > threshold).astype(int)
y_val_pred_gru = (y_val_pred_prob_gru > threshold).astype(int)
y_test_pred_gru = (y_test_pred_prob_gru > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_gru = accuracy_score(y_train_resampled, y_train_pred_gru)
validation_accuracy_gru = accuracy_score(y_val, y_val_pred_gru)
testing_accuracy_gru = accuracy_score(y_test, y_test_pred_gru)

training_classification_rep_gru = classification_report(y_train_resampled, y_train_pred_gru)
validation_classification_rep_gru = classification_report(y_val, y_val_pred_gru)
testing_classification_rep_gru = classification_report(y_test, y_test_pred_gru)

# Save the metrics using the provided function
save_accuracies("GRU", training_accuracy_gru, validation_accuracy_gru, testing_accuracy_gru,
                training_classification_rep_gru, validation_classification_rep_gru, testing_classification_rep_gru)

# Print the metrics
print("GRU - Training Accuracy:", training_accuracy_gru)
print("GRU - Training Classification Report:\n", training_classification_rep_gru)
print("\nGRU - Validation Accuracy:", validation_accuracy_gru)
print("GRU - Validation Classification Report:\n", validation_classification_rep_gru)
print("\nGRU - Testing Accuracy:", testing_accuracy_gru)
print("GRU - Testing Classification Report:\n", testing_classification_rep_gru)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_gru = confusion_matrix(y_train_resampled, y_train_pred_gru)
confusion_matrix_val_gru = confusion_matrix(y_val, y_val_pred_gru)
confusion_matrix_test_gru = confusion_matrix(y_test, y_test_pred_gru)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_gru, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('GRU - Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_gru, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('GRU - Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_gru, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('GRU - Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
