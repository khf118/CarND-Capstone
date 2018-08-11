from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self):

        # self.SSD_GRAPH_FILE = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/SSD_Mobilenet/Sim/frozen_inference_graph.pb'
        self.SSD_GRAPH_FILE = 'light_classification/SSD_Mobilenet/Sim/frozen_inference_graph.pb'
        self.confidence_cutoff = 0.6

        #self.SSD_GRAPH_FILE = '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/SSD_Inception_v2/Sim/frozen_inference_graph.pb'
        #self.confidence_cutoff = 0.7

        #load classifier
        self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def load_graph(self,graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self,min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self,boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #from styx_msgs/TrafficLight:
        #uint8 UNKNOWN=4
        #uint8 GREEN=2
        #uint8 YELLOW=1
        #uint8 RED=0

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(self.confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            #width, height = image.size
            #box_coords = to_image_coords(boxes, height, width)

            if boxes.size == 0:
                return TrafficLight.UNKNOWN

            # todo: get best result based on either size of box, or highest probability
            # HERE NEED TO SORT CLASSES INTO BEST RESULT FIRST (BASED ON PROB OR BOX SIZE ETC.)

            if(classes[0] == 2):
                return TrafficLight.RED

            if(classes[0] == 1):
                return TrafficLight.GREEN

            if(classes[0] == 3):
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
