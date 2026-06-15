############################
# turn_to_point algorithms #
############################

def turn_to_point_turn_ms(angle: float) -> int:
    """Determine how many milliseconds to turn for based on the angle to the point to which the robot is turning"""
    return max(100, min(500, int(abs(angle) * 5)))

def turn_to_point_turn_speed(angle: float) -> int:
    """Determine the speed with which to turn based on the angle to the point to which the robot is turning"""
    return max(30, min(100, int(abs(angle) * 0.4)))

############################
# drive_forward algorithms #
############################

def drive_forward_ms(distance: float) -> int:
    """Determine for how many milliseconds to drive forwards based on the distance to the target"""
    return max(100, min(2000, int(distance * 5)))

def drive_forward_speed(distance: float) -> int:
    """Determine the speed with which to drive forwards based on the distance to the target"""
    return max(30, min(100, int(distance * 0.8)))
