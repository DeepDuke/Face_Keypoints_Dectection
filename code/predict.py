"""
predict using trained model,
draw predicted landmarks on 112*112 input image and
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from network_utils import MyNet, MyNet2, MyDataSet
from PIL import Image


def predict(trained_model_path, model, loader, data):
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            image = batch['image']
            landmarks = batch['landmarks']  # 1*1*42

            landmarks_truth = landmarks.numpy()[0, 0, :]
            print('len of landmarks_truth: ', len(landmarks_truth))
            x = list(map(int, landmarks_truth[0:: 2]))
            y = list(map(int, landmarks_truth[1:: 2]))
            landmarks_truth = list(zip(x, y))

            output = model(image)
            output = output.numpy()[0]
            output_x = list(map(int, output[0:: 2]))
            output_y = list(map(int, output[1:: 2]))
            landmarks_predicted = list(zip(output_x, output_y))
            # print('landmarks_predicted:', landmarks_predicted)
            # draw on 112*112
            image = image.numpy()[0].astype(np.uint8)
            print('image shape: ', image.shape)
            image = image.transpose(1, 2, 0)  # torch.Tensor C*H*W, numpy: H*W*C
            print('image shape: ', image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for landmark_truth, landmark_predicted in zip(landmarks_truth, landmarks_predicted):
                # green truth landmarks
                cv2.circle(image, center=tuple(landmark_truth), radius=2, color=(0, 255, 0), thickness=-1)
                # blue predicted landmarks
                cv2.circle(image, center=tuple(landmark_predicted), radius=2, color=(255, 0, 0), thickness=-1)

            cv2.imshow('112', image)
            cv2.imwrite('..\\Result\\' + str(batch_idx)+'.jpg', image)

            # draw on original size image
            origin_image_path = data[batch_idx][0]
            x1, y1 = int(float(data[batch_idx][1])), int(float(data[batch_idx][2]))  # rect left up point
            x2, y2 = int(float(data[batch_idx][3])), int(float(data[batch_idx][4]))  # rect right down point
            rect_w = x2-x1
            rect_h = y2-y1
            origin_image = Image.open(origin_image_path).convert('RGB')
            origin_image = np.asarray(origin_image, dtype=np.uint8)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
            for landmark_truth, landmark_predicted in zip(landmarks_truth, landmarks_predicted):
                # green truth landmarks
                x, y = landmark_truth
                x = x1 + x / 112 * rect_w
                y = y1 + y / 112 * rect_h
                cv2.circle(origin_image, center=(int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
                # blue predicted landmarks
                x, y = landmark_predicted
                x = x1 + x / 112 * rect_w
                y = y1 + y / 112 * rect_h
                cv2.circle(origin_image, center=(int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)
            cv2.imshow('original image', origin_image)
            cv2.imwrite('..\\Result\\origin' + str(batch_idx)+'.jpg', origin_image)
            key = cv2.waitKey()
            if key == 27:
                # exit()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = '..\\mature\\model1.pt'
    # model_path = '..\\mature\\model2.pt'
    test_txt_path = '..\\test.txt'
    my_data = []
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()  # list
        for line in lines:
            line = line.split(' ')
            my_data.append(line)
    torch.manual_seed(1)
    use_cuda = False  # torch.cuda.is_available()
    my_device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_set = MyDataSet(test_txt_path, 'test')
    test_loader = DataLoader(test_set, **kwargs)

    my_model = MyNet().to(my_device)
    # prediction
    predict(model_path, my_model, test_loader, my_data)

