BALLS_PER_DELIVERY = 4 # how many balls to collect before delivery

BALL_INTAKE_ON_FOR_SECONDS = 500
BALL_INTAKE_SPEED = 100

EJACULATE_SPEED = 100

#
# nudge_robot constants
#

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

TURN_TO_POINT_SLEEP_BUFFER = 0.05
"""After sending the instruction to turn, sleep for the turn time plus this buffer"""
