import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from IPython import display
import numpy as np

plt.ion()

def plot(ctr, total_ctr):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    fig, ax = plt.subplots()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Product Recommendation in Online Advertising with Reinforcement Learning')
    #plt.title('Agent training result')
    plt.xlabel('Episode')
    plt.ylabel('Click-Through Rate (CTR)')
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    # ax.yaxis.set_ticks(np.arange(0, 2.0, 0.25))    
    plt.plot(ctr)
    plt.plot(total_ctr)
    plt.text(len(total_ctr) - 1, total_ctr[-1], "{:.2f}".format(total_ctr[-1]))
    plt.show(block = False)
    plt.pause(.1)

def plot_loss(loss):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Product Recommendation in Online Advertising with Reinforcement Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error (MSE)')
    plt.plot(loss)
    plt.show(block = False)
    plt.pause(.1)

def save_plot_image(name='agent_training.png'):
    plt.savefig(name)
