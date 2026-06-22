###########################
# Arena tracker constants #
###########################

ARUCO_OFFSET_X = 12
"""The absolute value of the amount of offset applied to the aruco maker to place it on its turning axis"""

######################################
# start_autonomous_session constants #
######################################

BALLS_PER_DELIVERY = 4
"""How many balls to collect before delivering"""

ROBOT_TO_POINT_DISTANCE_BEFORE_BURST = 10 + ARUCO_OFFSET_X
"""The distance within which the robot must be of the robot before bursting forwards to collect the ball"""

BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS = 10
"""The amount of seconds after which an estimate of ball count is invalid and must be re-estimated"""

WIN_MESSAGE = "I did it"
"""Message said aloud by the robot when finishing"""

SLEEP_BUFFER_SECONDS = 0.05
"""When sleeping after sending an instruction, sleep for the instruction's duration plus this buffer."""

GO_TO_BALL_APPROACH_RADIUS = 5.0 + ARUCO_OFFSET_X
"""Approach radius for go_to for normal balls"""

GO_TO_BALL_EDGE_APPROACH_RADIUS = 12 + ARUCO_OFFSET_X
"""Approach radius for go_to for edge balls"""

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
# burst_backward constants #
############################

BACKWARD_SPEED = 50
"""The speed at which to drive backwards"""

BACKWARD_MS = 500
"""The amount of milliseconds to drive backwards"""

#############################
# burst_into_ball constants #
#############################

BURST_FORWARD_MS = 450
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

DRIVE_TO_CENTER_DISTANCE_TOLERANCE = 10 + ARUCO_OFFSET_X
"""The range of the target point within which the robot must be before beginning driving towards the goal"""

###########################
# drive_to_goal constants #
###########################

DRIVE_TO_GOAL_AT_GOAL_RANGE = 17 + ARUCO_OFFSET_X
"""When driving to goal, the robot is considered to be at the goal when within this distance"""

DRIVE_TO_GOAL_CLOSE_TO_GOAL_RANGE = 25 + ARUCO_OFFSET_X
"""When driving to goal, the robot is considered to be close to the goal when within this distance"""

GOAL_DELIVERY_POINT = (152.5, 121.5/2)
"""The point where the robot is considered at the goal for deliveries"""

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

CROSS_WAYPOINT_OFFSET = CROSS_AVOIDANCE_WAYPOINTS_OFFSET + 9
"""The distance from center of cross to the corner of the waypoint zone. Distance is the avoidance
    zone plus size of cross"""

#######################################
# get_cross_approach_points constants #
#######################################

CROSS_APPROACH_POINTS_VERTICAL_OFFSET = 5

CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET = 10

###########################
# edge handling constants #
###########################

EDGE_THRESHOLD = 13
"""If a ball is within this distance of an edge, it is considered an edge ball"""

#############################
# corner handling constants #
#############################

CORNER_THRESHOLD = 24
"""If a ball is within this distance of a corner measured along the wall that the ball is adjacent to, it is considered a corner ball"""

BACK_TOWARDS_EDGE_MAX_ITERATIONS = 7
"""The maximum amount of times the controller can attempt backing towards a wall before giving up on the corner ball"""

WALL_THRESHOLD = 14
"""The range of the edge wherein the robot is considered to be at the edge"""

CORNER_BALL_COLLECTION_APPROACH_RADIUS = 10
"""Approach radius used in go_to when collecting corner ball"""

CORNER_COLLECTION_STAGING_POINT = (51, 18)
"""Stating point for south-west corner balls adjacent to the west wall"""

#####################
# heading constants #
#####################

EAST_HEADING = 0.0

NORTH_EAST_HEADING = 45.0

NORTH_HEADING = 90.0

NORTH_WEST_HEADING = 135.0

WEST_HEADING = 180.0

SOUTH_WEST_HEADING = 225.0

SOUTH_HEADING = 270.0

SOUTH_EAST_HEADING = 315.0

########################
# arena size constants #
########################

ARENA_WIDTH_CM = 167.0
"""Width of the arena in cm"""

ARENA_HEIGHT_CM = 121.5
"""Height of the arena in cm"""

######################################
# Robot constants #
######################################

ROBOT_LENGTH = 24
"""The length of the robot"""

ROBOT_WIDTH = 15
"""The width of the robot"""

DISTANCE_TO_POINT_TO_WITHIN_TURNING_HIT_RADIUS = 19
"""The distance to point to be considered within turning hit radius of the robot"""

DISTANCE_TO_ANGLE_TO_WITHIN_TURNING_HIT_RADIUS = 23
"""The angle to point to be considered within turning hit radius of the robot"""

LENGTH_OF_BOX_BEHIND_ROBOT = 18
"""The length of the box behind the robot"""

DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_BACK = ROBOT_TO_POINT_DISTANCE_BEFORE_BURST + 2.5

DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_FRONT = ROBOT_TO_POINT_DISTANCE_BEFORE_BURST - 5
