import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Plot the loss
def plot_loss(figure_title:str,
              loss_collector:list,
              loss_max_at_the_end:float,
              history:dict,
              save:bool=False,
              save_path_dir:str='',
              show:bool=True,
              ):

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    if 'recon_loss' in history.history.keys():
        ax.plot(history.history['recon_loss'])
        ax.plot(history.history['val_recon_loss'])
        ax.plot(history.history['kl_loss'])
        ax.plot(history.history['val_kl_loss'])
        plt.legend(['train loss',
                    'val loss',
                    'recon_loss',
                    'val recon_loss',
                    'kl_loss',
                    'val kl_loss'
                    ],
                   loc='upper right'
                   )
        figure_title = figure_title + '_detailed'
    else:
        plt.legend(['train loss', 'val loss'], loc='upper right')
    ax.ylim(min(loss_collector) * 0.9, loss_max_at_the_end * 2.0)
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    if save:
        plt.savefig(save_path_dir + "results/" + figure_title + '.png')
    if show:
        plt.show()
    plt.close()
    return fig, ax