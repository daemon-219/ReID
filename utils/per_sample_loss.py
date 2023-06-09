import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

def plot_loss_distribution(real_loss, fake_loss, save_path=''):
    plt.clf()
    ax = plt.gca()

    # font = {'family': 'Times New Roman',
    #          'weight': 'normal',
    #          'size': 13,
    #          }
    font1 = {
                'weight': 'normal',
                'size': 13,
            }

    # Plot data histogram
    ax.hist(real_loss, bins=100, density=True, histtype='stepfilled', color='darkorchid', alpha=0.3,
            label='Real samples')
    ax.hist(fake_loss, bins=100, density=True, histtype='stepfilled', color='blue', alpha=0.3,
            label='Fake samples')  

    ax.set_xlabel('Loss', fontdict=font1)
    ax.set_ylabel('Frequency', fontdict=font1)
    x_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=11)
    ax.legend(loc='upper left', prop=font1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    
    batch_size = len(real_loss)
    real_prob =  get_prob(np.array(real_loss+fake_loss))
    acc = ((real_prob[:batch_size]>=0.5).sum() + (real_prob[batch_size:]<0.5).sum()) / (batch_size*2)
    return acc


def get_prob(X):
    X = X.reshape(-1, 1)
    gmm = GMM(n_components=2).fit(X)
    predict_proba = gmm.predict_proba(X)
    means = gmm.means_
    # set the distribution with lower mean to be real 
    confidence = predict_proba[:,0] if means[0] < means[1] else predict_proba[:,1]

    return confidence