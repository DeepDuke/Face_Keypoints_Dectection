"""
Compute L1Loss on test dataset
"""
from network_utils import MyDataSet, MyNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def compute(trained_model_path, model, criterion, device, loader):
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    test_mean_points_loss = 0.0
    test_batch_num = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            test_img, landmarks = batch['image'], batch['landmarks']

            input_img = test_img.to(device)
            target_pts = landmarks.to(device)

            output_pts = model(input_img)

            test_loss = criterion(target_pts, output_pts)

            test_mean_points_loss += test_loss.item()
            test_batch_num += 1
    test_mean_points_loss /= test_batch_num
    print('\n' + '*' * 80 + '\n')
    print('test_mean_pts_loss: {:.6f}'.format(test_mean_points_loss))
    print('\n' + '*' * 80 + '\n')


if __name__ == '__main__':
    model_path = '..\\mature\\epoch 100 bs 128 lr e-3 ratio 0 L1 3.23 3.11\\detector_epoch_99.pt'
    # model_path = '..\\mature\\epoch 100 bs 128 lr e-3 ration 0 22 21\\detector_epoch_99.pt'
    test_txt_path = '..\\test.txt'
    use_cuda = True  # torch.cuda.is_available()
    my_device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_set = MyDataSet(test_txt_path, 'test')
    test_loader = DataLoader(test_set, **kwargs)

    my_model = MyNet().to(my_device)
    my_criterion = nn.L1Loss().to(my_device)
    # my_criterion = nn.MSELoss().to(my_device)

    compute(model_path, my_model, my_criterion, my_device, test_loader)

