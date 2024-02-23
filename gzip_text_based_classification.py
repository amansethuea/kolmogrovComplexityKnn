import gzip
import csv


class TextClassification(object):

    # Constructor function to have dataset available throughout the class scope
    def __init__(self):
        self.text_dataset = self.read_dataset()

    # Reading questionnaire dataset and storing it as a tuple in a list
    def read_dataset(self):
        text_dataset = []
        with open('dataset.csv', 'r', newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)
            for line in csv_reader:
                question = line[0]
                answer = line[1]
                text_dataset.append((question, answer))
            f.close()
        return text_dataset

    def calculate_ncd(self, x_1, x_2):
        # Compression of c(x_1) and c(x_2) strings separately
        c_x1 = len(gzip.compress(x_1.encode()))
        c_x2 = len(gzip.compress(x_2.encode()))
        x1_x2 = " ".join([x_1, x_2])
        # Compression of concatenated strings c(x_1 + x_2)
        c_x1_x2 = len(gzip.compress(x1_x2.encode()))
        # min of (c_x1, c_x2) defines smallest compressed size between x_1 and x_2
        # max of (c_x1, c_x2) defines largest compressed size between x_1 and x_2
        natural_compressed_distance = (c_x1_x2 - min(c_x1, c_x2)) / max(c_x1, c_x2)
        return natural_compressed_distance

    # This function predicts the class for a given input question using KNN but without training data
    def knn_classifier_without_train(self, ques, dataset, k=3):
        nearest_distances = []
        # Calculating NCD for all the questions in the dataset
        for (question, answer) in dataset:
            distance = self.calculate_ncd(ques, question)
            nearest_distances.append((distance, answer))
        nearest_distances.sort(key=lambda x: x[0])
        # Top classes as per k value
        # Stores the neighbours as per value of k having the best / smallest NCD values to the input text / question
        k_classes = [nearest_distances[i][1] for i in range(k)]
        print(nearest_distances)
        print(k_classes)
        # Get the first out of the 3 classes answer
        predicted_class = k_classes[0]
        return predicted_class

    def main(self, input_question, k=3):
        print(self.knn_classifier_without_train(input_question, self.text_dataset, k=k))


if __name__ == "__main__":
    obj = TextClassification()
    obj.main("What is the capital of France?")
    print()
    obj.main("Who developed the theory of relativity?")
