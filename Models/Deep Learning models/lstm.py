from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Create an LSTM model
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=X_train_rnn.shape[1:], return_sequences=True),
    Dropout(0.5),
    LSTM(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_timer()
lstm_model.fit(X_train_rnn, y_train_resampled, epochs=40, batch_size=32, verbose=1)
training_time = stop_timer("LSTM")

# Predict probabilities
y_train_pred_prob_lstm = lstm_model.predict(X_train_rnn)
y_val_pred_prob_lstm = lstm_model.predict(X_val_rnn)
y_test_pred_prob_lstm = lstm_model.predict(X_test_rnn)

# Convert probabilities to binary predictions using a threshold
threshold = 0.5
y_train_pred_lstm = (y_train_pred_prob_lstm > threshold).astype(int)
y_val_pred_lstm = (y_val_pred_prob_lstm > threshold).astype(int)
y_test_pred_lstm = (y_test_pred_prob_lstm > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_lstm = accuracy_score(y_train_resampled, y_train_pred_lstm)
validation_accuracy_lstm = accuracy_score(y_val, y_val_pred_lstm)
testing_accuracy_lstm = accuracy_score(y_test, y_test_pred_lstm)

training_classification_rep_lstm = classification_report(y_train_resampled, y_train_pred_lstm)
validation_classification_rep_lstm = classification_report(y_val, y_val_pred_lstm)
testing_classification_rep_lstm = classification_report(y_test, y_test_pred_lstm)

# Save the metrics using the provided function
save_accuracies("LSTM", training_accuracy_lstm, validation_accuracy_lstm, testing_accuracy_lstm,
                training_classification_rep_lstm, validation_classification_rep_lstm, testing_classification_rep_lstm)

# Print the metrics
print("LSTM - Training Accuracy:", training_accuracy_lstm)
print("LSTM - Training Classification Report:\n", training_classification_rep_lstm)
print("\nLSTM - Validation Accuracy:", validation_accuracy_lstm)
print("LSTM - Validation Classification Report:\n", validation_classification_rep_lstm)
print("\nLSTM - Testing Accuracy:", testing_accuracy_lstm)
print("LSTM - Testing Classification Report:\n", testing_classification_rep_lstm)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_lstm = confusion_matrix(y_train_resampled, y_train_pred_lstm)
confusion_matrix_val_lstm = confusion_matrix(y_val, y_val_pred_lstm)
confusion_matrix_test_lstm = confusion_matrix(y_test, y_test_pred_lstm)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_lstm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('LSTM - Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_lstm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('LSTM - Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_lstm, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('LSTM - Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()