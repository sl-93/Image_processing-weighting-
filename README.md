In this project, I created an application to detect labels among some productions and extract the weights of them. 

First, I tuned the Yolov5 for object detection based on my dataset.
Then, I used easyocr to extract texts from the labels.
After some image processing I extracted the exact weight from each label and summed them all.

I have used "Labelimg" for creating my dataset suitable for Yolo.

I have created my UI with PyQt:

![image](https://github.com/user-attachments/assets/b0ea057e-f25b-4254-993e-22b3da5f328e)

![image](https://github.com/user-attachments/assets/4ddeda01-d5f4-4d03-a498-8d10054a4573)


