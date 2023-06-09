import torch
from torch.nn import CrossEntropyLoss
from torch.nn.modules import loss
from utils.TripletLoss import TripletLoss
from utils.per_sample_loss import get_prob


class Loss(loss._Loss):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum
    
class Loss_triplet(loss._Loss):
    def __init__(self):
        super(Loss_triplet, self).__init__()

    def forward(self, outputs, labels):
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        print('\rTriplet_Loss:%.2f' % (
            Triplet_Loss.data.cpu().numpy()),
              end=' ')
        return Triplet_Loss
    
class Loss_CE(loss._Loss):
    def __init__(self):
        super(Loss_CE, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss()

        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        print('\rCrossEntropy_Loss:%.2f' % (
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return CrossEntropy_Loss


class Reweighted_Loss(loss._Loss):
    def __init__(self):
        super(Reweighted_Loss, self).__init__()

    def forward(self, outputs, labels):
        cross_entropy_loss = CrossEntropyLoss(reduction='none')
        triplet_loss = TripletLoss(margin=1.2)

        Triplet_Loss = [triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        # confidence only for CE
        CrossEntropy_Loss = [cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)
        real_prob = torch.from_numpy(get_prob(CrossEntropy_Loss.cpu().detach().numpy())).cuda()
        CrossEntropy_Loss = (real_prob * CrossEntropy_Loss).mean()

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum