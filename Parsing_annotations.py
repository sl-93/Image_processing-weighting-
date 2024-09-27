'''
This file parses annotations which are in the format of xml and create the dataset in the format of csv.
'''

import os
import xml.etree.ElementTree as ET
import glob
from enum import Enum

class Parameters(Enum):
    IMAGE_DIR = 'images'
    ANNOTATION_DIR = 'annotations'
    LABEL_DIR = 'labels'
    CLASSES = ['code']

class CreateDataset:
    def __init__(self, image_dir, annotation_dir, label_file):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.label_dir = label_file

    def convert_annotation(self, xml_file, output_dir):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find('filename').text
        img_path = os.path.join(self.image_dir, image_name)

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        output_file = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')

        with open(output_file, 'w') as out_file:
            for obj in root.findall('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in Parameters.CLASSES.value or int(difficult) == 1:
                    continue
                cls_id = Parameters.CLASSES.value.index(cls)

                xmlbox = obj.find('bndbox')
                x_min = int(xmlbox.find('xmin').text)
                x_max = int(xmlbox.find('xmax').text)
                y_min = int(xmlbox.find('ymin').text)
                y_max = int(xmlbox.find('ymax').text)


                x_center = (x_min + x_max) / (2.0 * width)
                y_center = (y_min + y_max) / (2.0 * height)
                bbox_width = (x_max - x_min) / float(width)
                bbox_height = (y_max - y_min) / float(height)

                out_file.write(f"{cls_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")


    def process_directory(self, split):
        annotation_path = os.path.join(self.annotation_dir, split)
        label_path = os.path.join(self.label_dir, split)

        os.makedirs(label_path, exist_ok=True)

        for xml_file in glob.glob(f"{annotation_path}/*.xml"):
            self.convert_annotation(xml_file, label_path)


if __name__ == "__main__":
    Dataset = CreateDataset(Parameters.IMAGE_DIR.value, Parameters.ANNOTATION_DIR.value, Parameters.LABEL_DIR.value)
    Dataset.process_directory('new')
    # Dataset.process_directory('validation')