import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import random


def channel_norm(img):
    # img: ndarray
    mean = np.mean(img)
    std = np.std(img)
    eps = 0.0000001
    pixels = (img - mean) / (std + eps)
    return pixels


class Normalize(object):
    def __call__(self, sample):
        """
        Normalize a img
        :param sample: {'image': PIL.Image, 'landmarks': np.ndarray}
        :return: normalized sample
        """
        img, landmarks = sample['image'], sample['landmarks']
        img = np.asarray(img, dtype=np.float32)
        # img = channel_norm(img)
        return {'image': img,
                'landmarks': landmarks}


class HorizontalFlip(object):
    def __call__(self, sample):
        """
        Flip a image horizontally with a certain probability
        :param sample: {'image': np.ndarray, 'landmarks': np.ndarray}
        :return: flipped sample
        """
        img, landmarks = sample['image'], sample['landmarks']
        p = random.random()
        if p <= 1:
            h, w, c = img.shape
            for i in range(w//2):
                img[:, i, :], img[:, w-1-i, :] = img[:, w-1-i, :], img[:, i, :]
            for i in range(0, len(landmarks[0]), 2):
                x = landmarks[0][i]
                landmarks[0][i] = w-1-x
        return {'image': img,
                'landmarks': landmarks}


class ChannelShuffle(object):
    def __call__(self, sample):
        """
        Shuffle the RGB channels randomly
        :param sample: {'image': numpy.ndarray, 'landmarks': numpy.ndarray}
        :return: sample after shuffle the RGB channels
        """
        image, landmarks = sample['image'], sample['landmarks']
        choices = ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                   (1, 2, 0), (2, 1, 0), (2, 0, 1))
        p = random.random()
        if p <= 0.5:
            idx = random.randint(0, 5)
            swap = choices[idx]
            image = image[:, :, swap]
        return {'image': image,
                'landmarks': landmarks}


class ToTensor(object):
    def __call__(self, sample):
        """
        convert a img to torch.tensor
        :param sample: {'image': np.ndarray, 'landmarks': np.ndarray}
        :return: processed sample
        """
        img, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        # img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        return {'image': img,
                'landmarks': landmarks}


transforms_train = transforms.Compose(
    [
        Normalize(),
        # HorizontalFlip(),
        # ChannelShuffle(),
        ToTensor()
    ]
)
transforms_test = transforms.Compose(
    [
        Normalize(),
        ToTensor()

    ]
)


class MyDataSet(Dataset):
    def __init__(self, data_path, mode):
        """
        :param data_path: path for train.txt or valid.txt or test.txt
        :param mode: mode for transforms
        """
        with open(data_path, 'r') as f:
            lines = f.readlines()
            self.data = lines
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx: sample index
        :return: {'image': PIL.Image, 'landmarks': np.ndarray}
        """
        line = self.data[idx]
        line = line.strip('\n').split(' ')
        img_path = line[0]
        line[1:] = list(map(float, line[1:]))  # convert str to float
        x1, y1, x2, y2 = line[1:5]
        img = Image.open(img_path).convert('RGB')
        img_crop = img.crop(tuple([int(x1), int(y1), int(x2), int(y2)]))  # crop image
        width, height = img_crop.size
        img_crop = img_crop.resize((112, 112), Image.BILINEAR)  # resize image

        landmarks = line[5:]
        assert len(landmarks) == 42, 'error: length of landmarks does not equal to 42 !'
        for i in range(0, len(landmarks), 2):  # resize landmarks
            x_idx, y_idx = i, i+1
            assert isinstance(landmarks[x_idx], float) and isinstance(landmarks[y_idx], float), \
                'landmarks are not float variable !!!'
            landmarks[x_idx] = (landmarks[x_idx] - x1) / width * 112
            landmarks[y_idx] = (landmarks[y_idx] - y1) / height * 112
        landmarks = np.asarray([landmarks]).astype(np.float32)
        sample = {'image': img_crop, 'landmarks': landmarks}
        if self.mode == 'train':
            sample = transforms_train(sample)
        elif self.mode == 'valid' or self.mode == 'test':
            sample = transforms_test(sample)

        return sample


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # Backbone:
        # in_channel, out_channel, kernel_size, stride, padding
        # block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=0)
        # block 2
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        # block 3
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        # block 4
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        # points branch
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.ip1 = nn.Linear(80 * 4 * 4, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        # common used
        self.prelu = nn.PReLU()
        self.avg_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape should be 1x3x112x112:', x.shape)
        x = self.avg_pool(self.prelu(self.conv1_1(x)))
        # print('b1: after block1, shape should be 1x8x27x27:', x.shape)
        # block 2
        x = self.prelu(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu, shape should be 1x16x25x25:', x.shape)
        x = self.prelu(self.conv2_2(x))
        # print('b2: after conv2_2 and prelu, shape should be 1x16x23x23:', x.shape)
        x = self.avg_pool(x)
        # print('x: after block2 and pool shape should be 1x16x12x12: ', x.shape)
        # block 3
        x = self.prelu(self.conv3_1(x))
        # print('b3: after conv3_1 and prelu shape should be 1x24x10x10: ', x.shape)
        x = self.prelu(self.conv3_2(x))
        # print('b3: after conv3_2 and prelu shape should be 1x24x8x8: ', x.shape)
        x = self.avg_pool(x)
        # print('x after block3 and pool shape should be 1x24x4x4: ', x.shape)
        # block 4
        x = self.prelu(self.conv4_1(x))
        # print('x after conv4_1 and prelu shape should be 1x40x4x4: ', x.shape)
        x = self.prelu(self.conv4_2(x))
        # print('x after conv4_2 and prelu shape should be 1x80x4x4: ', x.shape)
        # points branch
        ip = x.view(-1, 4 * 4 * 80)
        # print('ip flatten shape should be 1x1280: ', ip.shape)
        ip = self.prelu(self.ip1(ip))
        # print('ip shape after ip1 should be 1x128: ', ip.shape)
        ip = self.prelu(self.ip2(ip))
        # print('ip shape after ip2 should be 1x128: ', ip.shape)
        ip = self.ip3(ip)
        # print('ip shape after ip3 should be 1x42: ', ip.shape)

        return ip


class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        # Backbone:
        self.Backbone = nn.Sequential(
            # Backbone block1
            nn.Conv2d(3, 32, 3, 1, 0),  # 110*110
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 0),  # 108 * 108
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),  # kernel_size=2, stride=1, padding=0, 54*54
            # Backbone block2
            nn.Conv2d(64, 64, 3, 1, 1),  # 54*54
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),  # 54*54
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),  # 27*27
            # Backbone block3
            nn.Conv2d(128, 128, 3, 1, 1),  # 27*27
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),  # 27*27
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),  # 27*27
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),  # 14*14
            # Backbone block4
            nn.Conv2d(256, 256, 3, 1, 1),  # 14*14
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),  # 14*14
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),  # 14*14
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),  # 7*7
            # Backbone block5
            nn.Conv2d(256, 256, 3, 1, 1),  # 7*7
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),  # 7*7
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),  # 7*7
            nn.PReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True)  # 4*4
        )
        self.Branch = nn.Sequential(
            nn.Linear(256*4*4, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 42)
        )
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Backbone
        x = self.Backbone(x)
        x = x.view(-1, 256*4*4)
        x = self.Branch(x)
        return x


if __name__ == '__main__':
    train_data_path = '..\\train.txt'
    train_mode = 'train'
    train_data = MyDataSet(train_data_path, train_mode)
    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for idx, batch in enumerate(train_data_loader):
        img, landmarks = batch['image'], batch['landmarks']
        img = img[0].numpy()  # NxCxHxW ==> CxHxW
        img = img.transpose((1, 2, 0)).astype('uint8')  # CxHxW ==> HxWxC
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = landmarks.numpy()[0][0]  # 1x1x42 ==> 1x42
        pts_x = landmarks[0::2]
        pts_y = landmarks[1::2]
        for x, y in zip(pts_x, pts_y):
            cv2.circle(img, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)
        # print('img shape: ', img.shape)
        # print('landmarks shape: ', landmarks.shape)
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()

