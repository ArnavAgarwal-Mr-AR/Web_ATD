#Since the distribution in data is non-uniform, we will use the MinMax Scaling technique, and SMOTE is the chosen oversampling technique.

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Separate features and target
X = df.drop(['stroke'], axis=1)
y = df['stroke']

# Split data into train and test sets (80% train + validation, 20% test)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split train+validation into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.20, random_state=42)

# Apply feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

#Checking the distribution after oversampling on the training data:
from matplotlib import pyplot as plt
import seaborn as sns

y_resampled = y_train_resampled  # Use the resampled target values

plt.figure(figsize=(5, 3))
y_resampled.value_counts().plot.pie(autopct="%1.2f%%", colors=sns.color_palette('Set2'), explode=[0, 0.12], title="Training data Class Distribution After SMOTE")
plt.ylabel('stroke')
plt.show()


