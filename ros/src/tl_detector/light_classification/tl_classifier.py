from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self,path):

        self.SSD_GRAPH_FILE = path
        self.confidence_cutoff = 0.6


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

    def get_strong_classifications(self, minimum_score, boxes, scores, classes):
        """return list of identifications that have a score equal or
        higher than min score (drop all others)"""
        
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= minimum_score:
                idxs.append(i)

        good_boxes = boxes[idxs, ...]
        good_scores = scores[idxs, ...]
        good_classes = classes[idxs, ...]
        
        return good_boxes, good_scores, good_classes

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
            # run detection using the neural network
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # trim unused dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # trim out identified targets that have a classification probabilty less than confidence_cutoff
            boxes, scores, classes = self.get_strong_classifications(self.confidence_cutoff, boxes, scores, classes)

            if boxes.size == 0:
                return TrafficLight.UNKNOWN

            # todo: get best result based on either size of box, or highest probability
            # HERE NEED TO SORT CLASSES INTO BEST RESULT FIRST (BASED ON PROB OR BOX SIZE ETC.)

            # by default the calssifier returns the highest probability detection first so we will take that as our detected light state

            if(classes[0] == 2):
                return TrafficLight.RED

            if(classes[0] == 1):
                return TrafficLight.GREEN

            if(classes[0] == 3):
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
