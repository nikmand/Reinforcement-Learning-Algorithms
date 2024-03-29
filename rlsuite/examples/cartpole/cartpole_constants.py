
environment = 'CartPole-v0'

# 2 ή 4, 2, 6, 2
x_freq = 4
x_dot_freq = 2
theta_freq = 6
theta_dot_freq = 2

var_freq = [x_freq, x_dot_freq, theta_freq, theta_dot_freq]

USE_TENSORBOARD = True
LOG_WEIGHTS = False
LOGGER_PATH = 'logging.conf'

max_episodes = 500
max_steps = 195

# epsilon
# EPS_START = 1.0
# EPS_END = 0.01
# EPS_DECAY = 0.001  # drops to min around 100 episodes # 0.0005

EVAL_INTERVAL = 10
TERM_INTERVAL = 100 // EVAL_INTERVAL


def out_of_bounds(cartpole_x_pos):
    return True if cartpole_x_pos < -2.4 or cartpole_x_pos > 2.4 else False


def check_termination(eval_durations):
    return sum(list(eval_durations.values())[-TERM_INTERVAL:]) / TERM_INTERVAL >= 195
