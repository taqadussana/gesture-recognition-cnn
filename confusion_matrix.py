import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

MODEL_PATH = "gesture_model_final.h5"
DATASET_PATH = "gesture_dataset"
IMG_SIZE = 128
BATCH_SIZE = 32

CLASSES = ["none", "ok", "stop", "thumbs_up"]

model = tf.keras.models.load_model(MODEL_PATH)

val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in val_data:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cmap='Blues'
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Gesture Recognition")
plt.show()
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure()
plt.bar(CLASSES, class_accuracy)
plt.xlabel("Gesture Class")
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy of Gesture Recognition")
plt.ylim(0, 1)
plt.show()
