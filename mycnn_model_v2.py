import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, BatchNormalization, Dropout
from keras import utils
from keras import callbacks
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


## Load dataset
y = np.load('Y_train_new.npy', allow_pickle=True)
X = np.load('X_train_new.npy', allow_pickle=True)

# Check if Y is already one-hot encoded
if len(y.shape) == 2 and y.shape[1] > 1:
    Y = y  # Assume it's already one-hot encoded
else:
    # Encode class labels
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    Y = utils.to_categorical(encoded_Y, np.unique(encoded_Y).size)

# Stratified K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
fold_no = 0
cv_results = []

for train_idx, test_idx in kfold.split(X, np.argmax(Y, axis=1)):
    print(f"\nTraining on Fold {fold_no}...\n")

    # Split dataset into training and testing
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Reshape data for Conv1D
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create model
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Y.shape[1], activation='softmax'))

    # Compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # Early stopping to prevent overfitting
    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=20, min_delta=0.001, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train, Y_train, batch_size=32, epochs=100,  # Reduce epochs for each fold
        validation_data=(X_test, Y_test), callbacks=[earlystopping], verbose=1
    )

    # Save validation accuracy
    val_acc = max(history.history['val_accuracy'])
    cv_results.append(val_acc)

    print(f"Fold {fold_no} - Validation Accuracy: {val_acc:.4f}")
    model_filename = f"model_fold{fold_no}.h5"
    model.save(model_filename)
    print(f"✅ Model for Fold {fold_no} saved as '{model_filename}'")
    ##
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'learning_curve_{fold_no}.png')  # Save the figure
    plt.show()  # Display the figure
    ##
    fold_no += 1

# Print final cross-validation result
print("\nCross-Validation Results:")
print(f"Mean Accuracy: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")
