############################
# turn_to_point algorithms #
############################

def turn_to_point_turn_ms(angle: float) -> int:
    """Determine how many milliseconds to turn for based on the angle to the point to which the robot is turning"""
    return max(100, min(1500, int(abs(angle) * 10)))


def turn_to_point_turn_speed(angle: float) -> int:
    """Determine the speed with which to turn based on the angle to the point to which the robot is turning"""
    return max(30, min(100, int(abs(angle) * 0.5)))

############################
# drive_forward algorithms #
############################

def drive_forward_ms(distance: float) -> int:
    """Determine for how many milliseconds to drive forwards based on the distance to the target"""
    if distance < 15:
        forward_ms = 100
    elif distance < 20:
        forward_ms = 200
    elif distance < 30:
        forward_ms = 300
    elif distance < 40:
        forward_ms = 400
    else:
        forward_ms = int(min(100 + (6/5) ** (2 / 3 * distance), 1500))

    return forward_ms


    return int(min(100 + (6/5) ** (2 / 3 * distance), 1500))

def drive_forward_speed(distance: float) -> int:
    """Determine the speed with which to drive forwards based on the distance to the target"""
    return max(30, min(100, int(distance * 1.0)))
