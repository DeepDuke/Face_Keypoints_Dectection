"""
predict using trained model
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from network_utils import MyNet, MyDataSet
from PIL import Image


def predict(trained_model_path, model, loader, data):
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            image = batch['image']
            landmarks = batch['landmarks']  # 1*1*42
            # print('landmarks: ', landmarks)
            # truth
            landmarks_truth = landmarks.numpy()[0, 0, :]
            # print('landmarks numpy: ', landmarks_truth)
            print('len of landmarks_truth: ', len(landmarks_truth))
            x = list(map(int, landmarks_truth[0: len(landmarks_truth): 2]))
            y = list(map(int, landmarks_truth[1: len(landmarks_truth): 2]))
            landmarks_truth = list(zip(x, y))
            # print('landmarks_truth: ', landmarks_truth)
            # prediction

            output = model(image)
            output = output.numpy()[0]
            # print('output: ', output)
            output_x = list(map(int, output[0: len(output): 2]))
            output_y = list(map(int, output[1: len(output): 2]))
            landmarks_predicted = list(zip(output_x, output_y))
            # print('landmarks_predicted:', landmarks_predicted)
            # draw on 112*112
            image = image.numpy()[0]
            print('image shape: ', image.shape)
            image = image.transpose(1, 2, 0)  # torch.Tensor C*H*W, numpy: H*W*C
            print('image shape: ', image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for landmark_truth, landmark_predicted in zip(landmarks_truth, landmarks_predicted):
                # green truth landmarks
                cv2.circle(image, center=tuple(landmark_truth), radius=3, color=(0, 255, 0), thickness=-1)
                # blue predicted landmarks
                cv2.circle(image, center=tuple(landmark_predicted), radius=3, color=(255, 0, 0), thickness=-1)

            cv2.imshow(str(batch_idx), image)

            # draw on original size image
            origin_image_path = data[batch_idx][0]
            origin_image = Image.open(origin_image_path).convert('RGB')
            origin_w, origin_h = origin_image.size
            origin_image = np.asarray(origin_image, dtype=np.uint8)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
            for landmark_truth, landmark_predicted in zip(landmarks_truth, landmarks_predicted):
                # green truth landmarks
                x, y = landmark_truth
                x = x / 112 * origin_w
                y = y / 112 * origin_h
                cv2.circle(origin_image, center=(int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
                # blue predicted landmarks
                x, y = landmark_predicted
                x = x / 112 * origin_w
                y = y / 112 * origin_h
                cv2.circle(origin_image, center=(int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)
            cv2.imshow('original image', origin_image)
            # show truth image
            truth_image_name = data[batch_idx][0].split('\\')[-1]
            truth_image = Image.open(truth_image_name).convert('RGB')
            truth_image = np.asarray(truth_image, dtype='uint8')
            truth_image = cv2.cvtColor(truth_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('truth_image', truth_image)
            key = cv2.waitKey()
            if key == 27:
                # exit()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = '..\\trained_models\\detector_epoch_99.pt'
    test_txt_path = '..\\train.txt'
    data = []
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()  # list
        for line in lines:
            line = line.split(' ')
            data.append(line)
    torch.manual_seed(1)
    use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_set = MyDataSet(test_txt_path, 'test')
    test_loader = DataLoader(test_set, **kwargs)

    model = MyNet().to(device)
    # prediction
    predict(model_path, model, test_loader, data)
