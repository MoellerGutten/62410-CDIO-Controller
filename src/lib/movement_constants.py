######################################
# start_autonomous_session constants #
######################################

BALLS_PER_DELIVERY = 4 # how many balls to collect before delivery

ROBOT_TO_POINT_DISTANCE_BEFORE_BURST = 12 # the distance within which the robot must be of a ball before bursting forwards

BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS = 10 # amount of seconds after which an estimate of ball count is invalid and must be re-estimated

WIN_MESSAGE = "I did it"

SLEEP_BUFFER_SECONDS = 0.05
"""When sleeping after sending an instruction, sleep for the instruction's duration plus this buffer."""

################################
# _start_ball_intake constants #
################################

BALL_INTAKE_ON_FOR_SECONDS = 500
BALL_INTAKE_SPEED = 100

################################
# _start_ejaculation constants #
################################

EJACULATE_SPEED = 100

# #######################
# nudge_robot constants #
#########################

NUDGE_SPEED = 15
NUDGE_SECONDS = 0.3

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
