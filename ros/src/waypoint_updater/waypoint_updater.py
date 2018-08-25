#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Int32

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 40 # Number of waypoints we will publish. You can change this number
DECEL_LIMIT = -1
DIST_MARGIN = 3

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.sub_all_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        # TODO: Add other member variables you need below
        self.pose = None
        self.position = None
        self.heading_in_rad = None
        self.waypoints = None
        self.count = 0

        self.traffic_waypoint = None
        self.vel_base = 0.0
        self.current_vel = 0.0

        self.DECEL_RATE = 0.5

        self.time_old = rospy.get_time()
        self.time = 0
        self.stopping = False
        self.stop_dict = {}

        self.starting = True
        self.starting_idx = None
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.count += 1
            final_waypoints_exist = self.set_final_waypoints()

            if final_waypoints_exist:
                self.final_waypoints_pub.publish(self.final_waypoints)
            rate.sleep()


        rospy.spin()

    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x # m/s

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose
        # position
        self.position = self.pose.position
        # heading
        _x = self.pose.orientation.x
        _y = self.pose.orientation.y
        _z = self.pose.orientation.z
        _w = self.pose.orientation.w
        self.heading_in_rad = euler_from_quaternion([_x,_y,_z,_w])[2] # roll, pitch, yaw


    def waypoints_cb(self, waypoints):
        # TODO: Implement
        print " [*] GET WAYPOINTS ... "
        _max_vel = 0.0
        if self.waypoints == None:
            self.waypoints = []
            for idx, _waypoint in enumerate(waypoints.waypoints):
                _w = Waypoint()
                _w = _waypoint
                self.waypoints.append(_w)
                _vel = self.get_waypoint_velocity(_w)
                if _vel > _max_vel:
                    _max_vel = _vel
                    self.vel_base = self.get_waypoint_velocity(_w)


    def get_closest_waypoint(self):
        closest_dist = 999999.
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(len(self.waypoints)):
            dist = dl(self.position, self.waypoints[i].pose.pose.position)
            if dist < closest_dist:
                closest_dist = dist
                closest_waypoint = i
        return closest_waypoint

    def set_final_waypoints(self):
        final_waypoints_exist = False
        #  if (self.waypoints is not None) and (self.position is not None) and (self.traffic_waypoint is not None):
        if (self.waypoints is not None) and (self.position is not None):

            self.time = rospy.get_time()
            sample_time = self.time - self.time_old

            final_waypoints_exist = True

            # Get closest waypoint idx
            closest_waypoint = self.get_closest_waypoint()

            if self.traffic_waypoint is None:
                self.set_waypoint_velocity(self.waypoints, closest_waypoint, self.vel_base)
                self.stopping = False
            else:
                # Get index of stopline
                stopline_waypoint_idx = self.traffic_waypoint
                stopline_waypoint_idx = int(str(stopline_waypoint_idx).split()[1])

                # Get distance to next stopline
                dist_next_stopline = self.distance(self.waypoints, closest_waypoint, stopline_waypoint_idx)
                dist_next_stopline = max(0, dist_next_stopline - DIST_MARGIN)

                # Initialize
                req_a = 0
                stop_time = 0
                stop_distance = 0

                # If ahead traffic light is red, then calcultate
                # Request acceleration [m/s^2]
                # Expected time to stop [sec]
                # Expected distance to stop [m]
                if dist_next_stopline != 0:
                    req_a = -(self.current_vel ** 2) / (2 * dist_next_stopline)
                    stop_time = - (self.current_vel / req_a)
                    stop_distance = -(0.5 * req_a) * (stop_time ** 2) + (self.current_vel * stop_time)

                # Scheduling the target velocity from stop signal on position to stopline
                if req_a < -0.7 and stop_distance != 0 and stop_distance > dist_next_stopline and self.stopping == False and (stopline_waypoint_idx - closest_waypoint < 8):
                    self.stopping = True

                    waypoint_margin = 1
                    self.stop_dict = {}

                    # Get waypoint index to stopline and assign velocity to each waypoint
                    for i in range(0, stopline_waypoint_idx - closest_waypoint):

                        if stopline_waypoint_idx - closest_waypoint - waypoint_margin > 0:
                            vel_input = max(self.current_vel - 2 * (self.current_vel * (i+1) / (stopline_waypoint_idx - closest_waypoint - waypoint_margin)), 0)
                        else:
                            vel_input = 0.0

                        self.stop_dict[closest_waypoint + i] = vel_input
                #
                # # If vehicle req acceleration is less than max decel, than just go
                # if req_a < DECEL_LIMIT:
                #     self.stopping = False

                if stopline_waypoint_idx == -1:
                    # If green light
                    Is_red_light = False
                    if self.starting:
                        start_vel = self.vel_base / 4
                        for i in range(0,4):
                            self.set_waypoint_velocity(self.waypoints, closest_waypoint + i, start_vel + i * (self.vel_base / 4))
                        if self.starting_idx is None:
                            self.starting_idx = closest_waypoint + 4
                        if closest_waypoint >= self.starting_idx:
                            self.starting = False
                            self.starting_idx = closest_waypoint + 4
                    else:
                        self.set_waypoint_velocity(self.waypoints, closest_waypoint, self.vel_base)
                    self.stopping = False
                else:
                    Is_red_light = True
                    if self.stopping == True:
                        for i in range(0, stopline_waypoint_idx - closest_waypoint):
                            self.set_waypoint_velocity(self.waypoints, closest_waypoint + i, self.stop_dict[closest_waypoint + i])
                        self.starting = True
                    else:
                        if self.starting:
                            start_vel = self.vel_base / 4
                            for i in range(0,4):
                                self.set_waypoint_velocity(self.waypoints, closest_waypoint + i, start_vel + i * (self.vel_base / 4))
                            if self.starting_idx is None:
                                self.starting_idx = closest_waypoint + 4
                            if closest_waypoint >= self.starting_idx:
                                self.starting = False
                                self.starting_idx = closest_waypoint + 4
                        else:
                            self.set_waypoint_velocity(self.waypoints, closest_waypoint, self.vel_base)
                        self.stopping = False
                        #print(self.waypoints[closest_waypoint].twist.twist.linear.x)
                        

            # print('=======================================')
            # print('closest waypoint: ' + str(closest_waypoint))
            # print('stopline waypoint: ' + str(stopline_waypoint_idx))
            # print('current vel: ') + str(self.current_vel)
            # print('request acc: ' + str(req_a))
            # print('stop time: ' + str(stop_time))
            # print('stop distance: ' + str(stop_distance))
            # print('dist_next_stopline: ' + str(dist_next_stopline))
            # print('is stopping: ' + str(self.stopping))
            # print('Is_red_light: ' + str(Is_red_light))

            _lookahead_wps = LOOKAHEAD_WPS
            if closest_waypoint + _lookahead_wps > len(self.waypoints):
                _lookahead_wps = len(self.waypoints) - closest_waypoint

            # set final waypoints
            self.final_waypoints = Lane()
            self.final_waypoints.waypoints = []
            for i in range(_lookahead_wps):

                self.final_waypoints.waypoints.append(self.waypoints[closest_waypoint + i])

            self.time_old = self.time

        return final_waypoints_exist

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.traffic_waypoint = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
