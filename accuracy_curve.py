import pickle
import matplotlib.pyplot as plt

# Load saved history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

plt.figure()
plt.plot(history["accuracy"])
plt.plot(history["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"])
plt.show()
