In this project, I created an application to detect labels among some productions and extract the weights of them. 

First, I tuned the Yolov5 for object detection based on my dataset.
Then, I used easyocr to extract texts from the labels.
After some image processing I extracted the exact weight from each label and summed them all.

I have created my UI with PyQt.

I have used "Labelimg" for creating my dataset suitable for Yolo.
