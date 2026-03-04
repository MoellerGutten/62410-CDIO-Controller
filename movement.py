

# Right movements
def turn_right(degrees=None, times=1):
    if degrees is None:
        return "right," * (times-1) + "right"
    return f"right;{degrees}"

def turn_soft_right(degrees=None, times=1):
    if degrees is None:
        return "soft_right," * (times-1) + "soft_right"
    return f"soft_right;{degrees}"


# Left movements
def turn_left(degrees=None, times=1):
    if degrees is None:
        return "left," * (times-1) + "left"
    return f"left;{degrees}"

def turn_soft_left(degrees=None, times=1):
    if degrees is None:
        return "soft_left," * (times-1) + "soft_left"
    return f"soft_left;{degrees}"


# Forward movements
def drive_forward(distance=None):
    if distance is None:
        return "forward"
    return f"forward;{distance}"


# Backward movements
def drive_backward(distance=None):
    if distance is None:
        return "backward"
    return f"backward;{distance}"


def insert_instruction(instruction):
    with open('instructions.txt', 'a') as f:
        f.write(instruction)