import glob
from PIL import Image
import zlib
import os


class ImageClassification(object):
    def __init__(self):
        self.flag_path = []

    def calculate_ncd(self, x_1, x_2):
        # concatenation of image files
        x1_x2 = x_1 + x_2
        # compression of image file 1
        c_x1 = zlib.compress(x_1)
        # compression of image file 2
        c_x2 = zlib.compress(x_2)
        # Compression of concatenated images c(x_1 + x_2)
        c_x1_x2 = zlib.compress(x1_x2)
        # print len() of each file
        print(len(c_x1), len(c_x2), len(c_x1_x2), sep=' ', end='\n')
        # Calculation of Normalised Compressed Distance between two image files
        natural_compressed_distance = (len(c_x1_x2) - min(len(c_x1), len(c_x2))) / max(len(c_x1), len(c_x2))
        return natural_compressed_distance

    # Return all the paths of flag images
    def find_all_flag_images(self):
        for path_name in glob.glob(os.getcwd() + '/Flags/*.png'):
            self.flag_path.append(path_name)
        return self.flag_path

    # This function predicts the class for a given input image file using KNN but without training data
    def knn_classifier_without_train(self, input_x, k=3):
        data1 = Image.open(input_x)
        # Convert the image into bytes explicitly since zlib png compressor doesn't do it within itself
        data1 = data1.tobytes()
        images_dataset = self.find_all_flag_images()
        distances = []
        # Calculating NCD with input flag image (input_x) against every other flag image file
        for input_image_y in images_dataset:
            data2 = Image.open(input_image_y)
            data2 = data2.tobytes()
            distance = self.calculate_ncd(data1, data2)
            # Storing the NCD in a list
            distances.append((distance, input_image_y))
        distances.sort(key=lambda x: x[0])
        # Stores the neighbours as per value of k having the best / smallest NCD values to the input flag image
        k_classes = [distances[i][1] for i in range(k)]
        for dist in distances:
            print(dist)
        print(k_classes)
        predicted_class = k_classes[0]
        return predicted_class


if __name__ == "__main__":
    obj = ImageClassification()
    choose_flag = "nz.png"
    input_image_x = glob.glob(os.getcwd() + f"/Flags/{choose_flag}")[0]
    # nz flag is similar to australia and hm -> heard island amd mcdonald islands
    obj.knn_classifier_without_train(input_image_x, k=3)
