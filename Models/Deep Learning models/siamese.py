from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, classification_report

# Define the base network for Siamese Network
def base_network(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(16, activation='relu')
    ])
    return model

# Define Siamese Network
input_shape = X_train_resampled.shape[1:]
input_left = Input(shape=input_shape)
input_right = Input(shape=input_shape)

base_network_left = base_network(input_shape)
encoded_left = base_network_left(input_left)

base_network_right = base_network(input_shape)
encoded_right = base_network_right(input_right)

# Define distance function for the Siamese Network
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

distance_layer = Lambda(euclidean_distance)([encoded_left, encoded_right])

# Create Siamese model
siamese_model = Model(inputs=[input_left, input_right], outputs=distance_layer)

# Compile the model
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Start timer
start_timer()

# Train the model
siamese_model.fit([X_train_resampled, X_train_resampled], y_train_resampled, epochs=40, batch_size=32, verbose=1)

# Stop timer
training_time = stop_timer("Siamese Network")

# Evaluate on training, validation, and testing sets
y_train_pred_siamese = siamese_model.predict([X_train_resampled, X_train_resampled])
y_val_pred_siamese = siamese_model.predict([X_val_scaled, X_val_scaled])
y_test_pred_siamese = siamese_model.predict([X_test_scaled, X_test_scaled])

# Convert predictions to binary using a threshold
threshold = 0.5
y_train_pred_siamese_binary = (y_train_pred_siamese.round() > threshold).astype(int)
y_val_pred_siamese_binary = (y_val_pred_siamese.round() > threshold).astype(int)
y_test_pred_siamese_binary = (y_test_pred_siamese.round() > threshold).astype(int)

# Calculate and save the metrics
training_accuracy_siamese = accuracy_score(y_train_resampled, y_train_pred_siamese_binary)
validation_accuracy_siamese = accuracy_score(y_val, y_val_pred_siamese_binary)
testing_accuracy_siamese = accuracy_score(y_test, y_test_pred_siamese_binary)

training_classification_rep_siamese = classification_report(y_train_resampled, y_train_pred_siamese_binary)
validation_classification_rep_siamese = classification_report(y_val, y_val_pred_siamese_binary)
testing_classification_rep_siamese = classification_report(y_test, y_test_pred_siamese_binary)

# Save the metrics using the provided function
save_accuracies("Siamese Network", training_accuracy_siamese, validation_accuracy_siamese, testing_accuracy_siamese,
                training_classification_rep_siamese, validation_classification_rep_siamese, testing_classification_rep_siamese)

# Print the metrics
print("Siamese Network - Training Accuracy:", training_accuracy_siamese)
print("Siamese Network - Training Classification Report:\n", training_classification_rep_siamese)
print("\nSiamese Network - Validation Accuracy:", validation_accuracy_siamese)
print("Siamese Network - Validation Classification Report:\n", validation_classification_rep_siamese)
print("\nSiamese Network - Testing Accuracy:", testing_accuracy_siamese)
print("Siamese Network - Testing Classification Report:\n", testing_classification_rep_siamese)

# Print the training time
print("\nTraining Time:", training_time, "seconds")

#Plotting the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrices for training, validation, and testing sets
confusion_matrix_train_siamese = confusion_matrix(y_train_resampled, y_train_pred_siamese_binary)
confusion_matrix_val_siamese = confusion_matrix(y_val, y_val_pred_siamese_binary)
confusion_matrix_test_siamese = confusion_matrix(y_test, y_test_pred_siamese_binary)

# Define the labels for the confusion matrix
labels = ['No Stroke (0)', 'Stroke (1)']

# Create subplots for each confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training set confusion matrix
sns.heatmap(confusion_matrix_train_siamese, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Siamese Network - Training Set Confusion Matrix')
axes[0].set_xticklabels(labels)
axes[0].set_yticklabels(labels)

# Plot validation set confusion matrix
sns.heatmap(confusion_matrix_val_siamese, annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title('Siamese Network - Validation Set Confusion Matrix')
axes[1].set_xticklabels(labels)
axes[1].set_yticklabels(labels)

# Plot testing set confusion matrix
sns.heatmap(confusion_matrix_test_siamese, annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title('Siamese Network - Testing Set Confusion Matrix')
axes[2].set_xticklabels(labels)
axes[2].set_yticklabels(labels)

plt.tight_layout()
plt.show()
