######################################
# start_autonomous_session constants #
######################################

BALLS_PER_DELIVERY = 4
"""How many balls to collect before delivering"""

ROBOT_TO_POINT_DISTANCE_BEFORE_BURST = 12
"""The distance within which the robot must be of the robot before bursting forwards to collect the ball"""

BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS = 10
"""The amount of seconds after which an estimate of ball count is invalid and must be re-estimated"""

WIN_MESSAGE = "I did it"
"""Message said aloud by the robot when finishing"""

SLEEP_BUFFER_SECONDS = 0.05
"""When sleeping after sending an instruction, sleep for the instruction's duration plus this buffer."""

################################
# _start_ball_intake constants #
################################

BALL_INTAKE_ON_FOR_SECONDS = 500
"""The amount of seconds for which the ball collection motor will run before stopping when started"""

BALL_INTAKE_SPEED = 100
"""The speed with which the ball collection motor will run when ingesting balls"""

################################
# _start_ejaculation constants #
################################

EJACULATE_SPEED = 100
"""The speed with which the ball collection motor will run when spitting out balls"""

# #######################
# nudge_robot constants #
#########################

NUDGE_SPEED = 15
"""The speed with which the robot will be nudged when it is not identified and the controller is attempting to re-acquire it"""

NUDGE_SECONDS = 0.3
"""The amount of seconds for which the robot will be nudged when it is not identified and the controller is attempting to re-acquire it"""

###########################
# turn_to_point constants #
###########################

TURN_TO_POINT_PRECISE_TOLERANCE = 2.0
"""Tolerance used to determine if robot is facing point in precise mode"""

TURN_TO_POINT_TOLERANCE = 6.0
"""Tolerance used to determine if robot is facing point when not in precise mode"""

TURN_TO_POINT_PRECISE_SPEED = 10
"""Speed used when turning robot in precise mode"""

TURN_TO_POINT_PRECISE_MS = 100
"""Milliseconds to turn robot for in precise mode"""

############################
# drive_backward constants #
############################

BACKWARD_SPEED = 50
"""The speed at which to drive backwards"""

BACKWARD_MS = 500
"""The amount of milliseconds to drive backwards"""

#############################
# burst_into_ball constants #
#############################

BURST_FORWARD_MS = 250
"""The amount of miliseconds for which to burst forward when collecting a ball"""

BURST_FORWARD_SPEED = 75
"""The speed at which to burst forward when collecting a ball"""

###################
# go_to constants #
###################

GO_TO_DISTANCE_TOLERANCE = 10.0
"""The tolerance with which go_to determines whether the robot has reached the target point"""

GO_TO_MAX_MOVES = 20
"""The maximum amount of moves allowed in go_to"""

#############################
# drive_to_center constants #
#############################

DRIVE_TO_CENTER_DISTANCE_TOLERANCE = 10
"""The range of the target point within which the robot must be before beginning driving towards the goal"""

###########################
# drive_to_goal constants #
###########################

DRIVE_TO_GOAL_AT_GOAL_RANGE = 15
"""When driving to goal, the robot is considered to be at the goal when within this distance"""

DRIVE_TO_GOAL_CLOSE_TO_GOAL_RANGE = 25
"""When driving to goal, the robot is considered to be close to the goal when within this distance"""

###################################
# ball count estimation constants #
###################################

BALL_COUNT_ESTIMATION_SNAPSHOTS = 20
"""The amount of images to capture and analyze when estimating ball count"""

#########################
# await_robot constants #
#########################

MAX_ROBOT_DETECTION_ATTEMPTS = 5
"""The amount of attempts the controller will make to acquire the robot before attempting a nudge"""

#####################################
# Bounding box / waypoint constants #
#####################################

CROSS_AVOIDANCE_WAYPOINTS_OFFSET = 12
"""The amount of padding for the bounding box around the cross. Used for setting waypoints in the corners of the bounding box"""
