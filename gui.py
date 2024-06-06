import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Przeciągnij zdjęcie")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 600, 400)

        self.text_label = QLabel(self)
        self.text_label.setGeometry(50, 450, 300, 40)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            image_path = url.toLocalFile()
            pixmap = QPixmap(image_path)
            self.label.setPixmap(pixmap)
            self.label.setScaledContents(True)
            self.image_name = image_path.split('/')[-1]  # Zapisuje nazwę pliku
            result = self.my_function(image_path)  # Wykonuje swoją funkcję
            self.text_label.setText(result)  # Wyświetla tekst zwrócony przez funkcję

    def my_function(self, image_path):
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Rozszerzenie wymiarów o 1, aby uzyskać tensor (1, 150, 150, 3)
        img_array /= 255.0  # Przeskaluj obraz

        # Wykonaj predykcję na wczytanym obrazie
        prediction = loaded_model.predict(img_array)
        print(prediction)
        max_predict = 0
        index = 0
        for i, prob in enumerate(prediction[0]):
            if prob > max_predict:
                max_predict = prob
                index = i
        if index == 0:
            return "Przewidziano: KOT \npodobieństwo = " + str(max_predict)
        elif index == 1:
            return "Przewidziano: KROWA \npodobieństwo = " + str(max_predict)
        elif index == 2:
            return "Przewidziano: PIES \npodobieństwo = " + str(max_predict)
        elif index == 3:
            return "Przewidziano: KOŃ \npodobieństwo = " + str(max_predict)
        
if __name__ == "__main__":
    loaded_model = load_model("trained_model.h5")
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
