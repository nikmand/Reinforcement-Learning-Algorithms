import matplotlib.pyplot as plt
import configparser


def config_parser(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    # TODO add a warning if file is not found

    return config['env'], config['agent'], config['misc']


def init_tensorboard(launch_tensorboard):
    writer = None
    if launch_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    return writer


def log_parameters_histograms(tensorboard_writer, neural_net, i_episode, neural_net_name=None):
    """ Logs in Tensorboard histograms about weights and biases evolution during training."""

    for parameters_name, parameters_values in neural_net.named_parameters():
        layer_name, parameters_type = parameters_name.rsplit(".", 1)  # parameters_type: weights or biases
        if neural_net_name:
            neural_net_name += '/'
        tensorboard_writer.add_histogram(neural_net_name + layer_name + '/' + parameters_type, parameters_values,
                                         i_episode)
    tensorboard_writer.flush()

    return


def plot_epsilon(epsilon):
    """ Plot exploration probability per episode """
    plt.figure(2)
    plt.clf()
    plt.title('Epsilon...')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.plot(range(len(epsilon)), epsilon)
    plt.pause(0.001)


def plot_rewards(episode_durations, eval_durations):
    """ """
    figure = plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    train_i, train_duration = zip(*episode_durations.items())
    plt.plot(train_i, train_duration, label='train')
    try:
        eval_i, eval_duration = zip(*eval_durations.items())
        plt.plot(eval_i, eval_duration, marker=".", label='eval')
    except ValueError:
        pass
    plt.axhline(y=195, color='r')
    plt.legend(loc='best')
    plt.pause(0.001)  # pause a bit so that plots are updated

    return figure


def plot_rewards_completed(episode_durations, eval_durations):
    """  """
    fig = plot_rewards(episode_durations, eval_durations)
    plt.title('Progress after {} training episodes'.format(len(episode_durations)))

    return fig

