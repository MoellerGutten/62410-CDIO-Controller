

# Right movements
def turn_right(degrees=None, isMore=True):
    if degrees is None:
        return "right," if isMore else "right"
    return f"right;{degrees}," if isMore else f"right;{degrees}"

def turn_soft_right(degrees=None, isMore=True):
    if degrees is None:
        return "soft_right," if isMore else "soft_right"
    return f"soft_right;{degrees}," if isMore else f"soft_right;{degrees}"


# Left movements
def turn_left(degrees=None, isMore=True):
    if degrees is None:
        return "left," if isMore else "left"
    return f"left;{degrees}," if isMore else f"left;{degrees}"

def turn_soft_left(degrees=None, isMore=True):
    if degrees is None:
        return "soft_left," if isMore else "soft_left"
    return f"soft_left;{degrees}," if isMore else f"soft_left;{degrees}"


# Forward movements
def drive_forward(distance=None, isMore=True):
    if distance is None:
        return "forward," if isMore else "forward"
    return f"forward;{distance}," if isMore else f"forward;{distance}"


# Backward movements
def drive_backward(distance=None, isMore=True):
    if distance is None:
        return "backward," if isMore else "backward"
    return f"backward;{distance}," if isMore else f"backward;{distance}"
