import glob
import lzma
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
from PIL import Image


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
        for dist in distances:
            print(dist)
        get_ncd_with_class = [(distances[i][1], distances[i][0]) for i in range(k)]
        get_ncd_with_class.append((distances[0][1], distances[0][0]))
        return get_ncd_with_class

    def create_image_classification_graph(self, input_x, k=3):
        get_ncd_with_class = self.knn_classifier_without_train(input_x, k=k)
        # Function to create a graph for image classification using NCD
        G = nx.Graph()
        image_paths, ncds = [], []
        # Add nodes to the graph
        for image_ncd in get_ncd_with_class:
            image_paths.append(image_ncd[0])
            ncds.append(image_ncd[1])

        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                if ncds[i] <= 1.0:
                    G.add_edge(image_paths[i], image_paths[j], weight=ncds[i])
        return G

    def visualize_graph(self, graph):
        # Function to visualize the graph with images using matplotlib
        pos = nx.spring_layout(graph)

        # Create a plot
        fig, ax = plt.subplots()

        # Draw the graph
        nx.draw(graph, pos, ax=ax, with_labels=False, font_weight='bold', node_size=500, node_color='skyblue')
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        # Add images to nodes
        for node, (x, y) in pos.items():
            img = Image.open(node)
            img.thumbnail((300, 300))
            img_data = BytesIO()
            img.save(img_data, format="PNG")
            img_data.seek(0)
            img_array = plt.imread(img_data)
            imagebox = OffsetImage(img_array, zoom=0.1)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
            ax.add_artist(ab)

        plt.show()


if __name__ == "__main__":
    obj = ImageClassification()
    choose_flag = "nz.png"
    input_image_x = glob.glob(os.getcwd() + f"/Flags/{choose_flag}")[0]
    classification_graph = obj.create_image_classification_graph(input_image_x, k=3)
    obj.visualize_graph(classification_graph)