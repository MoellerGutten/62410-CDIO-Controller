############################
# turn_to_point algorithms #
############################

def turn_to_point_turn_ms(angle: float) -> int:
    """Determine how many milliseconds to turn for based on the angle to the point to which the robot is turning"""
    return max(100, min(500, int(abs(angle) * 5)))

def turn_to_point_turn_speed(angle: float) -> int:
    """Determine the speed with which to turn based on the angle to the point to which the robot is turning"""
    return max(30, min(100, int(abs(angle) * 0.4)))
