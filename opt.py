import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis', 'plot'],
                    help='train or evaluate ')

parser.add_argument('--fake_ratio',
                    default=0.0,
                    help='the ratio of fake image in training set',
                    type=float)

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    default=500,
                    help='number of epoch to train',
                    type=int)

parser.add_argument('--lr',
                    default=2e-4,
                    help='initial learning_rate',
                    type=float)

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='MultiStepLR, decay the learning rate')

parser.add_argument("--batchid",
                    default=4,
                    help='the batch for id',
                    type=int)

parser.add_argument("--batchimage",
                    default=4,
                    help='the batch of per id',
                    type=int)

parser.add_argument("--batchtest",
                    default=64,
                    help='the batch size for test',
                    type=int)

opt = parser.parse_args()
