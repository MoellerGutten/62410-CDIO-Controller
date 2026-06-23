############################
# turn_to_point algorithms #
############################

def turn_to_point_turn_ms(angle: float) -> int:
    """Determine how many milliseconds to turn for based on the angle to the point to which the robot is turning"""
    return max(100, min(750, int(abs(angle) * 6)))


def turn_to_point_turn_speed(angle: float) -> int:
    """Determine the speed with which to turn based on the angle to the point to which the robot is turning"""
    return max(30, min(100, int(abs(angle) * 0.4)))

############################
# drive_forward algorithms #
############################

def drive_forward_ms(distance: float) -> int:
    """Determine for how many milliseconds to drive forwards based on the distance to the target"""
    if distance < 15:
        forward_ms = 300
    elif distance < 20:
        forward_ms = 400
    elif distance < 30:
        forward_ms = 500
    elif distance < 40:
        forward_ms = 600
    else:
        forward_ms = int(min(100 + (6/5) ** (2 / 3 * distance), 2500))

    return forward_ms

def drive_forward_speed(distance: float) -> int:
    """Determine the speed with which to drive forwards based on the distance to the target"""
    return max(30, min(100, int(distance * 1.0)))

def burst_forward_ms(distance: float) -> int:
    """returns the time as a function of the distance to the target"""
    if distance < 12:
        burst_ms = 100
    elif distance < 14:
        burst_ms = 350
    else:
        burst_ms = 450

    return burst_ms

##################################################
# corner ball backwards staging point algorithms #
##################################################

def backward_ms(distance: float) -> int:
    return drive_forward_ms(distance)

def backward_speed(distance: float) -> int:
    return drive_forward_speed(distance)

