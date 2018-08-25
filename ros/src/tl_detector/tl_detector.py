#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import time
import threading

STATE_COUNT_THRESHOLD = 1

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        #sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size = 1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.model_path = self.config['model_path']
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.model_path)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.process_thread = None
        rate = rospy.Rate(15) # 50Hz
        while not rospy.is_shutdown():
            if self.camera_image is not None:
                #self.process_image()
                #t = threading.Thread(target=self.process_image)
                #t.start()
                if (self.process_thread is None or not self.process_thread.isAlive()):
                    self.process_thread = ImageProcess(self)
                    self.process_thread.start()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

class ImageProcess (threading.Thread):
    def __init__(self, tl_detector):
        threading.Thread.__init__(self)
        self.tl_detector = tl_detector

    def run(self):
        light_wp, state = self.process_traffic_lights()
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.tl_detector.state != state:
            self.tl_detector.state_count = 0
            self.tl_detector.state = state
        elif self.tl_detector.state_count >= STATE_COUNT_THRESHOLD:
            self.tl_detector.last_state = self.tl_detector.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.tl_detector.last_wp = light_wp
            self.tl_detector.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.tl_detector.upcoming_red_light_pub.publish(Int32(self.tl_detector.last_wp))
        self.tl_detector.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = 0
        closest_dist = float('inf')

        if self.tl_detector.waypoints:
            for wp_idx in range(len(self.tl_detector.waypoints)):
                distance = math.sqrt((pose.position.x-self.tl_detector.waypoints[wp_idx].pose.pose.position.x)**2 +
                                    (pose.position.y-self.tl_detector.waypoints[wp_idx].pose.pose.position.y)**2)

                if(distance < closest_dist):
                    closest_dist = distance
                    closest_idx = wp_idx

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.tl_detector.has_image):
            self.tl_detector.prev_light_loc = None
            return False

        cv_image = self.tl_detector.bridge.imgmsg_to_cv2(self.tl_detector.camera_image, "bgr8")

        # note, do we want to have this line? check if network trained on RGB or BGR - i think BGR
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        #Get classification
        return self.tl_detector.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        max_detection_dist = 120 # maximum distance we want to check lights for
        min_dist = float('inf') #closest light

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.tl_detector.config['stop_line_positions']

        if(self.tl_detector.pose and self.tl_detector.waypoints):
            car_position = self.get_closest_waypoint(self.tl_detector.pose.pose)

            # Find the closest visible traffic light (if one exists)
            for stop_pos in stop_line_positions:
                new_light = TrafficLight()

                #new_light.header = Header()
                new_light.header.stamp = rospy.Time.now()
                new_light.header.frame_id = 'world'

                new_light.pose.pose = Pose()
                new_light.pose.pose.position.x = stop_pos[0]
                new_light.pose.pose.position.y = stop_pos[1]

                new_light.state = TrafficLight.UNKNOWN

                stop_position = self.get_closest_waypoint(new_light.pose.pose)

                distance_to_light = math.sqrt((self.tl_detector.waypoints[car_position].pose.pose.position.x-self.tl_detector.waypoints[stop_position].pose.pose.position.x)**2 +
                                              (self.tl_detector.waypoints[car_position].pose.pose.position.y-self.tl_detector.waypoints[stop_position].pose.pose.position.y)**2)

                if distance_to_light < min_dist and distance_to_light < max_detection_dist: # if closer than last light, but not beyond max range we are interested in,
                    if car_position < stop_position: # and our car has not yet passed the wp the light is at, then...
                        min_dist = distance_to_light
                        light = new_light
                        light_wp = stop_position

        if light:
            state = self.get_light_state(light)
            return light_wp, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
