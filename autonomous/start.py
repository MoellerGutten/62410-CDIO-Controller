from model import state
from stateManager import update_state

def start_autonomous_session(state, frame, model, M, M_inv, logger):
    print("autonomous")
    update_state(state, frame, model, M, M_inv,logger)
    pass