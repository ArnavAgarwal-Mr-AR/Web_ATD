from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Reshape data for RNN (assuming sequential data like time series)
X_train_rnn = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
X_val_rnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
X_test_rnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Create an RNN model
rnn_model = Sequential([
    LSTM(64, activation='relu', input_shape=X_train_rnn.shape[1:], return_sequences=True),
    Dropout(0.5),
    LSTM(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_timer()
# Train the model
rnn_model.fit(X_train_rnn, y_train_resampled, epochs=40, batch_size=32, verbose=1)
training_time = stop_timer("RNN")

# Predict probabilities
y_train_pred_prob_rnn = rnn_model.predict(X_train_rnn)
y_val_pred_prob_rnn = rnn_model.predict(X_val_rnn)
y_test_pred_prob_rnn = rnn_model.predict(X_test_rnn)

# Convert probabilities to binary predictions using a threshold
threshold = 0.5
y_train_pred_rnn = (y_train_pred_prob_rnn > threshold).astype(int)
y_val_pred_rnn = (y_val_pred_prob_rnn > threshold).astype(int)
y_test_pred_rnn = (y_test_pred_prob_rnn > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_rnn = accuracy_score(y_train_resampled, y_train_pred_rnn)
validation_accuracy_rnn = accuracy_score(y_val, y_val_pred_rnn)
testing_accuracy_rnn = accuracy_score(y_test, y_test_pred_rnn)

training_classification_rep_rnn = classification_report(y_train_resampled, y_train_pred_rnn)
validation_classification_rep_rnn = classification_report(y_val, y_val_pred_rnn)
testing_classification_rep_rnn = classification_report(y_test, y_test_pred_rnn)

# Save the metrics using the provided function
save_accuracies("RNN", training_accuracy_rnn, validation_accuracy_rnn, testing_accuracy_rnn,
                training_classification_rep_rnn, validation_classification_rep_rnn, testing_classification_rep_rnn)

# Print the metrics
print("RNN - Training Accuracy:", training_accuracy_rnn)
print("RNN - Training Classification Report:\n", training_classification_rep_rnn)
print("\nRNN - Validation Accuracy:", validation_accuracy_rnn)
print("RNN - Validation Classification Report:\n", validation_classification_rep_rnn)
print("\nRNN - Testing Accuracy:", testing_accuracy_rnn)
print("RNN - Testing Classification Report:\n", testing_classification_rep_rnn)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confuion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_rnn = confusion_matrix(y_train_resampled, y_train_pred_rnn)
confusion_matrix_val_rnn = confusion_matrix(y_val, y_val_pred_rnn)
confusion_matrix_test_rnn = confusion_matrix(y_test, y_test_pred_rnn)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_rnn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_rnn, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_rnn, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()