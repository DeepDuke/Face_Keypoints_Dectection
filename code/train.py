from network_utils import MyDataSet, MyNet, MyNet2
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # create save model directory
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)
    epochs = args.epochs
    pts_criterion = criterion

    train_losses = []
    valid_losses = []

    for epoch_idx in range(epochs):
        # start training
        model.train()

        epoch_train_loss = 0.0
        epoch_batch_num = 0
        epoch_train_img_num = 0

        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmarks = batch['landmarks']

            # ground truth
            input_img = img.to(device)
            target_pts = landmarks.to(device)

            # clear all gradients of all optimized variables
            optimizer.zero_grad()

            # output
            output_pts = model(input_img)
            # batch mean train loss
            train_loss = pts_criterion(output_pts, target_pts)

            temp_loss = train_loss.item()
            epoch_train_loss += temp_loss
            epoch_batch_num += 1
            epoch_train_img_num += len(img)  # NxCxHxW

            # do BP
            train_loss.backward()
            optimizer.step()

            # print train log
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} Batch id: {} [{}/{} {:.0f}% \t {:.6f}]'.format(
                    epoch_idx,
                    batch_idx,
                    epoch_train_img_num,
                    len(train_loader.dataset),
                    100.0 * epoch_train_img_num / len(train_loader.dataset),
                    temp_loss
                ))
        # end train
        ################################################################################
        # start validating
        valid_mean_pts_loss = 0.0
        valid_batch_num = 0
        model.eval()
        with torch.no_grad():
            for valid_batch_idx, valid_batch in enumerate(valid_loader):
                valid_img = valid_batch['image']
                valid_landmarks = valid_batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = valid_landmarks.to(device)

                output_pts = model(input_img)

                valid_loss = pts_criterion(output_pts, target_pts)

                valid_mean_pts_loss += valid_loss
                valid_batch_num += 1
        # end valid
        ###############################################################
        # save loss
        valid_mean_pts_loss /= valid_batch_num
        train_mean_pts_loss = epoch_train_loss / epoch_batch_num
        print('\n' + '*' * 80 + '\n')
        print('train_mean_pts_loss: {:.6f}'.format(train_mean_pts_loss))
        print('valid_mean_pts_loss: {:.6f}'.format(valid_mean_pts_loss))
        print('\n' + '*' * 80 + '\n')
        train_losses.append(train_mean_pts_loss)
        valid_losses.append(valid_mean_pts_loss)

        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory,
                                                'detector_epoch' + '_' + str(epoch_idx) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
        # end epoch
    return train_losses, valid_losses


def main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch-size to train (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.995, metavar='M',
                        help='SGD momentum (default: 0.995)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable cuda in training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current model (default: true)')
    parser.add_argument('--save-directory', type=str, default='..\\trained_models',
                        help="learned model are saved here (default: '..\\trained_models')")
    args = parser.parse_args()

    ###########################################################################

    torch.manual_seed(args.seed)  # set random seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading data sets ...')
    train_data_path = '..\\train.txt'
    valid_data_path = '..\\valid.txt'
    train_data = MyDataSet(train_data_path, 'train')
    valid_data = MyDataSet(valid_data_path, 'valid')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, **kwargs)

    print('===> Loading model ...')
    model = MyNet().to(device)
    # model = MyNet2().to(device)
    # criterion_pts = nn.MSELoss().to(device)
    criterion_pts = nn.L1Loss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('===> Starting training ...')
    train_losses, valid_losses = \
        train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
    np.savetxt('train_losses.csv', train_losses, delimiter=',', fmt='%f')
    np.savetxt('valid_losses.csv', valid_losses, delimiter=',', fmt='%f')
    print('============================================================================')


if __name__ == '__main__':
    main()

