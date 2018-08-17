#!/usr/bin/env python

import numpy as np
import time
from scipy.optimize import minimize
import rospy
from styx_msgs.msg import Lane
from geometry_msgs.msg import PointStamped, TwistStamped
import tf
from math import atan, cos, sin, sqrt
import matplotlib.pyplot as plt

N = 5
dt = 0.1
n_vars = 6 * N 
# This is the length from front to CoG that has a similar radius.
Lf = 2.67

max_throttle = 1
max_angle = 0.43 * Lf

x_start = 0
y_start = x_start + N
psi_start = y_start + N
v_start = psi_start + N
cte_start = v_start + N
epsi_start = cte_start + N

delta_start = 0
a_start = N - 1

coeffs = None
ini_state = None
waypoints_x = None
#Return the state at each waypoint given a set of actuators
def get_state(x,ini_state):
	#new state
	state = np.zeros(n_vars)
	state[x_start] = ini_state[0]
	state[y_start] = ini_state[1]
	state[psi_start] = ini_state[2]
	state[v_start] = ini_state[3]
	state[cte_start] = ini_state[4]
	state[epsi_start] = ini_state[5]

	for t in range(1,N-1):
		x0 = state[x_start + t - 1]
		y0 = state[y_start + t - 1]
		psi0 = state[psi_start + t - 1]
		v0 = state[v_start + t - 1]
		cte0 = state[cte_start + t - 1]
		epsi0 = state[epsi_start + t - 1]

		delta0 = x[delta_start + t]
		a0 = x[a_start + t]
		
		f0 = np.polyval(coeffs,x0)
		psides0 = atan(3 * coeffs[3] * x0 * x0 + 2 * coeffs[2] * x0 + coeffs[1])

		state[x_start + t] = x0 + v0 * cos(psi0) * dt
		state[y_start + t] = y0 + v0 * sin(psi0) * dt
		state[psi_start + t] = psi0 - v0 * delta0 / Lf * dt
		state[v_start + t] = v0 + a0 * dt
		state[cte_start + t] = (f0 - y0) 
		state[epsi_start + t] = (psi0 - psides0) + v0 * delta0 / Lf * dt

	return state

def func(x,ini_state,coeffs):
	ref_v = 10

	state = get_state(x,ini_state)

	minimize = 0

	#for t in range(0, N-1): 
	#	to_minimize[0] += 10000 * pow(state[x_start + t] - waypoints_x[t], 2)
	
	for t in range(0, N-1):
		minimize += 3000 * pow(state[cte_start + t], 2)
		minimize += 300 * pow(state[epsi_start + t], 2)
		minimize += 1 * pow(state[v_start + t] - ref_v, 2)
	
	# Minimize the use of actuators.
	for t in range(0, N - 2):
		minimize += 5 * pow(x[delta_start + t], 2)
		minimize += 5 * pow(x[a_start + t], 2)
	
	#Minimize the value gap between sequential actuations.
	for t in range(0, N - 3):
		minimize += 100 * pow(x[delta_start + t + 1] - x[delta_start + t], 2)
		minimize += 10 * pow(x[a_start + t + 1] - x[a_start + t], 2)
	
	return minimize

def const_x_pos_min(actuations):
	state = get_state(actuations,ini_state)

	distances = np.zeros(N-1)
	for i in range(0,N-1):
		distances[i] = state[x_start+i] - waypoints_x[i]
	return np.sum(np.abs(distances)) 

def const_x_pos_max(actuations):
	state = get_state(actuations,ini_state)

	distances = np.zeros(N-1)
	for i in range(0,N-1):
		distances[i] = state[x_start+i] - waypoints_x[i]
	return 0.1 - np.sum(np.abs(distances))

class MPC:
	def __init__(self,mx):
#		rospy.init_node('mpc_node')
		self.state = np.array([])
		self.coeffs = np.array([])
		self.ini_state = None
		self.current_velocity = 10
		self.waypoints_x = []
		self.waypoints_y = []
		self.mx = mx
		print("inside")
#		waypoints_listner = rospy.Subscriber('/final_waypoints', Lane, self.wp_cb) 
		#TODO update Lane to correct msg type
#		velocity_listener = rospy.Subscriber('/current_velocity', TwistStamped, self.vel_cb) 

	def solve(self,state,coeffs):
		global ini_state
		global waypoints_x
		waypoints_x = self.waypoints_x
		ini_state = state
		#set model
		x = state[0]
		y = state[1]
		psi = state[2]
		v = state[3]
		cte = state[4]
		epsi = state[5]

		#setting bounds
		vars_lowerbound = np.zeros(2 * (N-1))
		vars_upperbound = np.zeros(2 * (N-1))

		for i in range(delta_start,a_start-1):
			vars_lowerbound[i] = -1 * max_angle
			vars_upperbound[i] = max_angle

		for i in range(a_start, a_start + N - 1):
			vars_lowerbound[i] = -1 * self.mx
			vars_upperbound[i] = self.mx

		bounds = np.array([])
		bounds = bounds.reshape((0,2))
		
		for index in range(0,2 * (N-1)):
			if vars_lowerbound[index] == 0:
				bounds = np.append(bounds,[(None,None)], axis = 0)
			else:
				bounds = np.append(bounds,[(vars_lowerbound[index],vars_upperbound[index])], axis = 0)
		
		#initial value
		x0 = np.zeros(2 * (N-1))

		x0 = x0.flatten()
		coeffs = coeffs.flatten()

		result = minimize(func, x0, args = (state,coeffs), bounds = bounds, constraints = [{"type":'ineq',"fun": const_x_pos_min}], method = "SLSQP", options = {"maxiter": 500, "disp": False})
		#print(result)
		#print("throttle ",result.x[a_start+1])
		return result.x

	def wp_cb(self,lane,current_velocity):
		global coeffs, dt
		#print("wp_cb")
		#current_waypoint= lane.waypoints[0]
		#transorming waypoint from world frame to vehicle frame
		waypoints_x = np.array([])
		waypoints_y = np.array([])

		listener = tf.TransformListener()
		listener.waitForTransform("/world", "/base_link", rospy.Time(0),rospy.Duration(4.0))
		
		for i in range(0,N-1):
			waypoint=PointStamped()
			waypoint.header.frame_id = "world"
			waypoint.header.stamp =rospy.Time(0)
			waypoint.point.x= lane.waypoints[i].pose.pose.position.x
			waypoint.point.y= lane.waypoints[i].pose.pose.position.y
			waypoint.point.z=0.0
			waypoint_base = listener.transformPoint("base_link",waypoint)
			waypoints_x = np.append(waypoints_x, waypoint_base.point.x)
			waypoints_y = np.append(waypoints_y, waypoint_base.point.y)
		
		state = np.zeros(6)
		dt = sqrt(pow(waypoints_y[1] - waypoints_y[0],2) + pow(waypoints_x[1] - waypoints_x[0],2)) / current_velocity
		print(current_velocity)
		print(dt)
		#waypoints_x = waypoints_x_w[0:N-1]
		#waypoints_y = waypoints_y_w[0:N-1]
		
		#for i in range(0,N-1):
		#	shift_x = waypoints_x[i] - waypoints_x[N-2];
		#	shift_y = waypoints_y[i] - waypoints_y[N-2];

		#	waypoints_x[i] = (shift_x * cos(-1 * psi) - shift_y * sin(-1 * psi));
		#	waypoints_y[i] = (shift_x * sin(-1 * psi) + shift_y * cos(-1 * psi));
        
		
		#plt.plot(waypoints_x,waypoints_y,"r")
		#plt.axis('equal')
		#plt.show()
		coeffs = np.polyfit(waypoints_x, waypoints_y, 3)
		#print(coeffs)
		images = np.array([])
		for i in range(0,N-1):
			images = np.append(images,np.polyval(coeffs,i))

		self.waypoints_x = np.array([i for i in range(0,N-1)])
		self.waypoints_y = images
		
		plt.plot(self.waypoints_x,self.waypoints_y,"b")
		plt.axis('equal')
		#plt.show()
		cte = np.polyval(coeffs, 0)
		epsi = - atan(coeffs[1])
		
		state = np.zeros(6)
		state[0] = 0
		state[1] = 0
		state[2] = 0
		state[3] = current_velocity
		state[4] = cte
		state[5] = epsi
		s = time.time()
		vars = self.solve(state,coeffs)
		#final_state = get_state(vars,state)
		print("Time :",time.time()-s)
		#np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
		#vars_x = final_state[x_start:x_start+N-1]
		#vars_y = final_state[y_start:y_start+N-1]
		#print("W_X ",np.array2string(self.waypoints_x, formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("W_Y ",np.array2string(self.waypoints_y, formatter={'float_kind':lambda x: "%.4f" % x}))
		
		#print("X ",np.array2string(final_state[x_start:x_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("Y ",np.array2string(final_state[y_start:y_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("PSI ",np.array2string(final_state[psi_start:psi_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("V ",np.array2string(final_state[v_start:v_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("CTE ",np.array2string(final_state[cte_start:cte_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("Epsi ",np.array2string(final_state[epsi_start:epsi_start+N-1], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("Delta ",np.array2string(vars[delta_start:delta_start+N-2], formatter={'float_kind':lambda x: "%.4f" % x}))
		#print("Throttle ",np.array2string(vars[a_start:a_start+N-2], formatter={'float_kind':lambda x: "%.4f" % x}))
		
		#print(vars_y.shape)
		#plt.show()
		#plt.plot(vars_x,vars_y,"g")
		#plt.axis('equal')
		#plt.show()

		return vars[delta_start],vars[a_start]
		
	def vel_cb(self,msg):
		self.current_velocity = msg.twist.linear.x
