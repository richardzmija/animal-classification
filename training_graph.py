import pandas as pd
import matplotlib.pyplot as plt


file_path = "training_process_data.csv"
training_data = pd.read_csv(file_path)

epochs = training_data["epoch"]
accuracy = training_data["accuracy"]
val_accuracy = training_data["val_accuracy"]

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, label="Training accuracy", marker="o")
plt.plot(epochs, val_accuracy, label="Validation accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_vs_validation_accuracy.png")
plt.show()
