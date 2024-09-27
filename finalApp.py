import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, \
    QStackedWidget, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
import torch
import pandas as pd
import re
import pickle
from threading import Thread
from PIL import Image
#-----------------------------requirement modlue------------------------------------------
#_________________________________________________________________________________________


def resource_path(relative_path):
    """ Get the absolute path to a resource, works for PyInstaller. """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller stores files in a temporary folder named _MEIPASS
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

MODEL_PATH = resource_path('models/model.pt')
OCR_PATH = resource_path('models/ocr_model.pickle')
IMAGE_PATH = resource_path("images/firstPage.jpg")


class OCR:
    def __init__(self, model_path, ocr_path, image_path):
        self.model_path = model_path
        self.ocr_path = ocr_path
        self.im = image_path

    def read_load(self):
        # load model
        self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
        # load ocr model
        with open(self.ocr_path, 'rb') as m:
            self.reader = pickle.load(m)

        # warmup
        im = str(self.im)

        # load the image for ocr model
        img = cv2.imread(self.im)

        # Run the model on the image
        self.model(im)
        self.reader.readtext(img)



    def run_model(self, image_path):

        self.image_path = image_path
        # load image for the model
        im = str(self.image_path)

        # load the image for ocr model
        img = cv2.imread(self.image_path)

        # Run the model on the image
        results = self.model(im)

        # Show the detected objects on the image
        # results.show()
        detections = results.pandas().xyxy[0]

        total = 0
        labels = []
        for idx, row in detections.iterrows():
            # Extract bounding box coordinates and class
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

            cropped_img = img[ymin:ymax, xmin:xmax]

            cropped_img = cv2.fastNlMeansDenoisingColored(cropped_img, h=3)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            (h, w) = cropped_img.shape[:2]
            if (h < 110 or w < 110):
                cropped_img = cv2.resize(cropped_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
            else:
                cropped_img = cv2.resize(cropped_img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)


            results = self.reader.readtext(cropped_img)
            x = pd.DataFrame(results, columns=["bbox", "text", "confidence"])

            for i, r in x.iterrows():

                r["text"] = r["text"].replace(',', '.')
                match1 = re.search(r'(\d+)/(\d+)\s*(g|gr|Gr)', r["text"])
                match2 = re.search(r'(W|w|WG):\s*(\d+)/(\d+)\s', r["text"])
                match3 = re.search(r'^\s*(\d+)/(\d+)\s*$', r["text"])
                match4 = re.search(r'(W|w|WG):(\d+)/(\d+)\s', r["text"])
                match8 = re.search(r'(W|w|WG)(\d+)/(\d+)', r["text"])


                match5 = re.search(r'(\d+)(\.)(\d+)\s*(g|gr|Gr)', r["text"])
                match6 = re.search(r'(W|w|WG):\s*(\d+)(\.)(\d+)', r["text"])
                match7 = re.search(r'(.+)(\.)(.+)', r["text"])
                match9 = re.search(r'(W|w|WG):(\d+)(\.)(\d+)', r["text"])
                match10 = re.search(r'(W|w|WG|Gr)(\d+\.\d+)', r["text"])

                if match1:
                    number = f"{match1.group(1)}.{match1.group(2)}"
                    number = float(number)
                    total += number
                    labels.append(number)
                    match1 = None

                if match2:
                    number = f"{match2.group(1)}.{match2.group(2)}"
                    number = float(number)
                    total += number
                    labels.append(number)
                    match2 = None

                if match3:
                    number = f"{match3.group(1)}.{match3.group(2)}"
                    number = float(number)
                    total += number
                    labels.append(number)
                    match4 = None

                if match4:
                    number = f"{match4.group(1)}.{match4.group(2)}"
                    number = float(number)
                    total += number
                    labels.append(number)
                    match3 = None

                if match8:
                    number = f"{match8.group(1)}.{match8.group(2)}"
                    number = float(number)
                    total += number
                    labels.append(number)
                    match8 = None


                if match5:
                    if match5.group(1).isdigit() and match5.group(3).isdigit():
                        f = re.findall('[-+]?(?:\d*\.*\d+)', r["text"])
                        number = float(f[0])
                        total += number
                        labels.append(number)
                    match5 = None

                if match6:
                    if match6.group(1).isdigit() and match6.group(3).isdigit():
                        f = re.findall('[-+]?(?:\d*\.*\d+)', r["text"])
                        number = float(f[0])
                        total += number
                        labels.append(number)
                    match6 = None

                if match7:
                    if match7.group(1).isdigit() and match7.group(3).isdigit():
                        f = re.findall('[-+]?(?:\d*\.*\d+)', r["text"])
                        number = float(f[0])
                        total += number
                        labels.append(number)
                    match7 = None

                if match9:
                        print(match9.group(1))
                        print(match9.group(2))
                        number = float(match9.group(2))
                        total += number
                        labels.append(number)
                        match9 = None

                if match10:
                        print(match10.group(1))
                        print(match10.group(2))
                        number = float(match10.group(2))
                        total += number
                        labels.append(number)
                        match10 = None


        return f"{total:.3f}", labels

class LoginPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        firstIm = QLabel()
        pixmap = QPixmap(resource_path('images/firstPage2.jpg'))
        scaled_pixmap = pixmap.scaled(700, 700, Qt.KeepAspectRatio)
        firstIm.setPixmap(scaled_pixmap)
        layout.addWidget(firstIm, alignment=Qt.AlignCenter)

        self.username_input = QLineEdit(self)
        self.username_input.setFixedSize(700, 40)
        self.username_input.setPlaceholderText(" Username")
        layout.addWidget(self.username_input, alignment=Qt.AlignCenter)


        self.password_input = QLineEdit(self)
        self.password_input.setFixedSize(700, 40)
        self.password_input.setPlaceholderText(" Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input, alignment=Qt.AlignCenter)

        #TODO: Enable the username and password
        self.username_input.setDisabled(True)
        self.password_input.setDisabled(True)

        login_button = QPushButton("Login", self)
        login_button.setFixedSize(700, 40)
        login_button.clicked.connect(self.check_login)
        layout.addWidget(login_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def check_login(self):
        self.stacked_widget.setCurrentIndex(1)

        # username = self.username_input.text()
        # password = self.password_input.text()
        # if username == "user" and password == "pass":
        #     self.stacked_widget.setCurrentIndex(1)
        # else:
        #     QMessageBox.warning(self, "Error", "Incorrect username or password!")

class MainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_image_path = None
        self.taken_image_path = None
        self.initUI()

    def initUI(self):
        layout1 = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setText("No Image Selected")
        self.image_label.setFixedSize(700, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout1.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.weight_label = QLabel(self)
        self.weight_label.setObjectName('weight')
        self.weight_label.setText("Total weight is ...")
        self.weight_label.setAlignment(Qt.AlignCenter)
        self.weight_label.setFixedSize(700, 110)
        layout1.addWidget(self.weight_label, alignment=Qt.AlignCenter)

        camera_layout = QHBoxLayout()
        self.capture_button = QPushButton("Take Photo", self)
        self.capture_button.setObjectName('cameraShotter')
        self.capture_button.setFixedSize(120, 120)
        self.capture_button.clicked.connect(self.capture_image)
        # self.capture_button.setEnabled(False)
        self.capture_button.hide()

        self.return_button = QPushButton('return', self)
        self.return_button.setObjectName('returnButton')
        self.return_button.setFixedSize(120, 120)
        self.return_button.clicked.connect(self.finish_webcam)
        camera_layout.addWidget(self.return_button)
        camera_layout.addWidget(self.capture_button)
        self.return_button.hide()

        layout1.addLayout(camera_layout)


        layout2 = QHBoxLayout()
        self.select_image_button = QPushButton("Select Image", self)
        self.select_image_button.setFixedSize(350, 50)
        self.select_image_button.clicked.connect(self.select_image)
        layout2.addWidget(self.select_image_button)

        self.take_photo_button = QPushButton("Start Webcam", self)
        self.take_photo_button.setFixedSize(350, 50)
        self.take_photo_button.clicked.connect(self.start_webcam)
        layout2.addWidget(self.take_photo_button)
        layout1.addLayout(layout2)

        self.setLayout(layout1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None


    def select_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.JPEG)",
                                                   options=options)
        self.weight_label.setText("Total weight is ...")

        if file_name:
            self.selected_image_path = file_name

            image = Image.open(file_name)

            try:
                exif = image._getexif()
                orientation = exif.get(274)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, TypeError):
                pass

            image_data = image.convert("RGBA").tobytes("raw", "RGBA")
            q_image = QImage(image_data, image.width, image.height, QImage.Format_RGBA8888)

            self.image_label.setPixmap(QPixmap.fromImage(q_image).scaled(700, 650, Qt.KeepAspectRatio))

            # weight = main.ocr.run_model(self.selected_image_path)
            # self.weight_label.setText(f"Total weight is {weight}")

            thread = Thread(target=self.run_model, args=(self.selected_image_path,), daemon=True)
            thread.start()

    def start_webcam(self):
        self.weight_label.setText("Total weight is ...")
        self.return_button.show()
        self.capture_button.show()
        self.capture_button.setEnabled(True)
        self.select_image_button.hide()
        self.take_photo_button.hide()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open webcam.")
            return
        self.timer.start(20)


    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.taken_image_path = "captured_image.png"
            cv2.imwrite(self.taken_image_path, frame)
            QMessageBox.information(self, "Photo Saved", f"Photo saved to {self.taken_image_path}")
            # weight = main.ocr.run_model(self.taken_image_path)
            # self.weight_label.setText(f"Total weight is {weight}")
            thread = Thread(target=self.run_model, args=(self.taken_image_path,), daemon=True)
            thread.start()

    def finish_webcam(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.image_label.setText("No Image Selected")
        self.capture_button.hide()
        self.select_image_button.show()
        self.take_photo_button.show()
        self.return_button.hide()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

    def run_model(self, image_path):
        self.weight_label.setText(f"is calculating ...")
        self.weight, labels = main.ocr.run_model(image_path)
        self.weight_label.setText(f"Total weight is {self.weight}\n{labels[:11]}\n{labels[11:]}")

class main(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Gold Scale')

        self.ocr = OCR(MODEL_PATH, OCR_PATH, IMAGE_PATH)
        self.ocr.read_load()

        self.login_page = LoginPage(self)
        self.main_page = MainPage()

        self.addWidget(self.login_page)
        self.addWidget(self.main_page)

        self.setFixedSize(800, 700)

    @staticmethod
    def apply_style(app):
        style = """
        QWidget {
            background-color: #1e1e1e;  
            color: #ffffff;  
            font-family: 'Arial';
        }
        
        QLabel {
            font-size: 16px;
            padding: 4px;
            border: 1px solid #ffffff;  
            border-radius: 10px;
            background-color: #2a2a2a;
        }
        
        QLabel#weight {
            font-size: 20px;
            padding: 3px;
            border: 5px solid #ffffff;  
            border-radius: 10px;
            background-color: #d4af37;
        }
        
        QPushButton {
            background-color: #d4af37;  
            color: #1e1e1e;
            font-size: 16px;
            padding: 5px;
            border-radius: 10px;
            border: 2px solid #d4af37;
        }
        
        QPushButton#returnButton {
            background-color: #000000;  
            color: #FFFFFF;
            font-size: 18px;
            padding: 5px;
            border-radius: 10px;
            border: 2px solid #d4af37;
            border-radius: 50px;  
            font-weight: bold;
        }
        
        QPushButton#cameraShotter {
            background-color: #000000;  
            color: #FFFFFF;
            font-size: 18px;
            padding: 5px;
            border-radius: 10px;
            border: 2px solid #d4af37;
            border-radius: 50px;  
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #ffcc00;  
            color: #1e1e1e;  
            border-color: #ffcc00;  
            font-size: 17px;  
            transform: scale(1.05);  
            transition: all 0.3s ease;  
        }
        
        QPushButton:disabled {
            background-color: #4a4a4a;  
            color: #a9a9a9;  
            border: 2px solid #4a4a4a;
            cursor: not-allowed;  
        }
        
        QLineEdit {
            background-color: #2a2a2a;
            color: #ffffff;
            border: 1px solid #d4af37;
            border-radius: 10px;
            padding: 8px;
        }
        
        QLineEdit:disabled {
            background-color: #4a4a4a;  
            color: #a9a9a9;  
            border: 1px solid #4a4a4a;
        }
        
        #webcam_label {
            border: 2px solid #d4af37;
            border-radius: 10px;
            margin-top: 10px;
        }
        
        QMessageBox {
            background-color: #2a2a2a;
            color: #ffffff;
        }


        """
        app.setStyleSheet(style)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = main()
    main.apply_style(app)
    main.show()

    sys.exit(app.exec_())


