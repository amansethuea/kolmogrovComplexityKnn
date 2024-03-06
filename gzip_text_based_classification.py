import gzip
import csv
import matplotlib.pyplot as plt
import numpy as np

class TextClassification(object):

    def __init__(self):
        self.text_dataset = self.read_dataset()

    def read_dataset(self):
        text_dataset = []
        with open('dataset.csv', 'r', newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # Skip header
            for line in csv_reader:
                question = line[0]
                answer = line[1]
                text_dataset.append((question, answer))
        return text_dataset

    def calculate_ncd(self, x_1, x_2):
        c_x1 = len(gzip.compress(x_1.encode()))
        c_x2 = len(gzip.compress(x_2.encode()))
        x1_x2 = " ".join([x_1, x_2])
        c_x1_x2 = len(gzip.compress(x1_x2.encode()))
        return (c_x1_x2 - min(c_x1, c_x2)) / max(c_x1, c_x2)

    def knn_classifier_without_train(self, ques, dataset, k=3):
        nearest_distances = []
        for (question, answer) in dataset:
            distance = self.calculate_ncd(ques, question)
            nearest_distances.append((distance, answer))
        nearest_distances.sort(key=lambda x: x[0])
        k_classes = [nearest_distances[i][1] for i in range(k)]
        predicted_class = max(set(k_classes), key=k_classes.count)
        return predicted_class

    def generate_confusion_matrix(self, test_data, k=3):
        tp = fp = tn = fn = 0
        for (question, true_answer) in test_data:
            predicted_answer = self.knn_classifier_without_train(question, self.text_dataset, k)
            if predicted_answer == true_answer:
                if predicted_answer == 'Yes':
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted_answer == 'Yes':
                    fp += 1
                else:
                    fn += 1
        return tp, fp, tn, fn

    def calculate_performance_metrics(self, tp, fp, tn, fn):
        # Calculate precision, recall, F1 score, and accuracy
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return precision, recall, f1_score, accuracy

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def main(self, test_dataset, k=3):
        tp, fp, tn, fn = self.generate_confusion_matrix(test_dataset, k)
        cm = np.array([[tp, fp], [fn, tn]])
        self.plot_confusion_matrix(cm, classes=['Yes', 'No'], title='Confusion Matrix')
        precision, recall, f1_score, accuracy = self.calculate_performance_metrics(tp, fp, tn, fn)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    obj = TextClassification()
    test_dataset = [("What is the capital of France?", "Yes"),  # Example test data; replace with actual data
                    ("Who developed the theory of relativity?", "No")]
    textDataset = obj.read_dataset();
    obj.main(textDataset)
