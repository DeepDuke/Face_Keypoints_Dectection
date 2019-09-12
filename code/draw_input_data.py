import cv2


label_path = '..\\train.txt'
# label_path = '..\\valid.txt'
# label_path = '..\\test.txt'
data = []
cnt_bad_points = 0
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

with open(label_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split(' ')
        line[1:] = list(map(float, line[1:]))
        data.append(line)
for line in data:
    img_path = line[0]
    img = cv2.imread(img_path)  # read image
    h, w, c = img.shape
    x1, y1, x2, y2 = line[1:5]  # rectangle points
    cv2.rectangle(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)),
                  color=(255, 0, 0), thickness=2)
    pts_x = line[5::2]  # x for all landmarks
    pts_y = line[6::2]  # y for all landmarks
    for x, y in zip(pts_x, pts_y):
        if x < x1 or y < y1 or x > x2 or y > y2:
            cnt_bad_points += 1
            print('bad point', (x, y))
            continue
        cv2.circle(img, center=(int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
    print('cnt_bad_points: ', cnt_bad_points)
    cv2.imshow('image', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()


