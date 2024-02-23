import glob
import lzma
import os


class ImageClassification(object):
    def __init__(self):
        self.flag_path = []

    def calculate_ncd(self, x_1, x_2):
        # Read image files to convert them into bytes
        x_1 = open(x_1, 'rb').read()
        x_2 = open(x_2, 'rb').read()
        # concatenation of image files
        x1_x2 = x_1 + x_2
        # compression of image file 1
        c_x1 = lzma.compress(x_1)
        # compression of image file 2
        c_x2 = lzma.compress(x_2)
        # Compression of concatenated images c(x_1 + x_2)
        c_x1_x2 = lzma.compress(x1_x2)
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
        images_dataset = self.find_all_flag_images()
        distances = []
        for input_image_y in images_dataset:
            # Calculating NCD with input flag image (input_x) against every other flag image file
            distance = self.calculate_ncd(input_x, input_image_y)
            # Storing the NCD in a list
            distances.append((distance, input_image_y))
        distances.sort(key=lambda x: x[0])
        # Stores the neighbours as per value of k having the best / smallest NCD values to the input flag image
        k_classes = [distances[i][1] for i in range(k)]
        print(distances)
        print(k_classes)
        predicted_class = k_classes[0]
        return predicted_class


if __name__ == "__main__":
    obj = ImageClassification()
    choose_flag = "nz.png"
    input_image_x = glob.glob(os.getcwd() + f"/Flags/{choose_flag}")[0]
    obj.knn_classifier_without_train(input_image_x, k=3)