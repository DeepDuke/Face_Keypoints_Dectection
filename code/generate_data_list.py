import random
from PIL import Image


dir_paths = ['..\\I', '..\\II']
data = []
for dir_path in dir_paths:
    label_path = dir_path + '\\' + 'label.txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(' ')
            line[0] = dir_path + '\\' + line[0]
            data.append(line)
            # print(line)
            # print(len(line))
    print(len(data))


# pre-process data
# TODO: convert str to float
# TODO: expand rectangle
for idx, line in enumerate(data):
    line[1:5] = list(map(float, line[1:5]))  # convert str to float
    img = Image.open(line[0]).convert('RGB')
    width, height = img.size
    ratio = 0  # 0.125  # expand padding ratio
    padding_width, padding_height = width*ratio, height*ratio

    x1, y1, x2, y2 = line[1:5]  # rectangle points
    x1 -= padding_width
    y1 -= padding_height
    x2 += padding_width
    y2 += padding_height

    line[1] = x1 if x1 > 0 else 0
    line[2] = y1 if y1 > 0 else 0
    line[3] = x2 if x2 < width else width-1
    line[4] = y2 if y2 < height else height-1

    assert (line[1] >= 0) and (line[2] >= 0) and (line[3] < width) and (line[4] < height), \
        'error at {}'.format(line)
    line[1:5] = list(map(str, line[1:5]))  # convert float back to str
    data[idx] = line  # save changes
    # print(width, height, line[1:5])
    # print(line)

# delete image containing bad points
data_filtered = []
for idx, line in enumerate(data):
    line[1:] = list(map(float, line[1:]))  # convert str to float
    x1, y1, x2, y2 = line[1:5]  # rectangle
    pts_x = line[5::2]
    pts_y = line[6::2]
    flag = True
    for x, y in zip(pts_x, pts_y):
        if x <= x1 or y <= y1 or x >= x2 or y >= y2:
            flag = False
            break
    if flag:
        line[1:] = list(map(str, line[1:]))
        data_filtered.append(line)

data = data_filtered
print('total {} images'.format(len(data)))
# generate train.txt, valid.txt, test.txt
for idx, line in enumerate(data):
    line = ' '.join(line)
    data[idx] = line
    assert isinstance(data[idx], str), '{} is not a str'.format(line)
    # print(line)

random.seed(1)
random.shuffle(data)  # shuffle data
train_data = data[0: int(len(data) * 0.8)]
valid_data = data[int(len(data) * 0.8): int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9):]

with open('..\\train.txt', 'w') as f:
    for line in train_data:
        assert isinstance(line, str), '{} is not a str'.format(line)
        # print('saving {} into train.txt'.format(line))
        f.write(line + '\n')

with open('..\\valid.txt', 'w') as f:
    for line in valid_data:
        assert isinstance(line, str), '{} is not a str'.format(line)
        # print('saving {} into valid.txt'.format(line))
        f.write(line + '\n')

with open('..\\test.txt', 'w') as f:
    for line in test_data:
        assert isinstance(line, str), '{} is not a str'.format(line)
        # print('saving {} into test.txt'.format(line))
        f.write(line + '\n')

print('\n' + '#'*66 + '\n')
print('===> Successfully saved {} images into train.txt'.format(len(train_data)))
print('===> Successfully saved {} images into valid.txt'.format(len(valid_data)))
print('===> Successfully saved {} images into test.txt'.format(len(test_data)))
print('\n' + '#'*66)
