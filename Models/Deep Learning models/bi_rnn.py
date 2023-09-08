from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Create a BiRNN model
birnn_model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=X_train_rnn.shape[1:]),
    Dropout(0.5),
    Bidirectional(LSTM(32, activation='relu')),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
birnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_timer()
birnn_model.fit(X_train_rnn, y_train_resampled, epochs=40, batch_size=32, verbose=1)
training_time = stop_timer("BiRNN")

# Predict probabilities
y_train_pred_prob_birnn = birnn_model.predict(X_train_rnn)
y_val_pred_prob_birnn = birnn_model.predict(X_val_rnn)
y_test_pred_prob_birnn = birnn_model.predict(X_test_rnn)

# Convert probabilities to binary predictions using a threshold
threshold = 0.5
y_train_pred_birnn = (y_train_pred_prob_birnn > threshold).astype(int)
y_val_pred_birnn = (y_val_pred_prob_birnn > threshold).astype(int)
y_test_pred_birnn = (y_test_pred_prob_birnn > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_birnn = accuracy_score(y_train_resampled, y_train_pred_birnn)
validation_accuracy_birnn = accuracy_score(y_val, y_val_pred_birnn)
testing_accuracy_birnn = accuracy_score(y_test, y_test_pred_birnn)

training_classification_rep_birnn = classification_report(y_train_resampled, y_train_pred_birnn)
validation_classification_rep_birnn = classification_report(y_val, y_val_pred_birnn)
testing_classification_rep_birnn = classification_report(y_test, y_test_pred_birnn)

# Save the metrics using the provided function
save_accuracies("BiRNN", training_accuracy_birnn, validation_accuracy_birnn, testing_accuracy_birnn,
                training_classification_rep_birnn, validation_classification_rep_birnn, testing_classification_rep_birnn)

# Print the metrics
print("BiRNN - Training Accuracy:", training_accuracy_birnn)
print("BiRNN - Training Classification Report:\n", training_classification_rep_birnn)
print("\nBiRNN - Validation Accuracy:", validation_accuracy_birnn)
print("BiRNN - Validation Classification Report:\n", validation_classification_rep_birnn)
print("\nBiRNN - Testing Accuracy:", testing_accuracy_birnn)
print("BiRNN - Testing Classification Report:\n", testing_classification_rep_birnn)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_birnn = confusion_matrix(y_train_resampled, y_train_pred_birnn)
confusion_matrix_val_birnn = confusion_matrix(y_val, y_val_pred_birnn)
confusion_matrix_test_birnn = confusion_matrix(y_test, y_test_pred_birnn)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_birnn, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_birnn, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_birnn, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
