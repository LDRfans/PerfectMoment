import numpy as np
import cv2
import dlib


def extract(img, dialation=0.25):
    '''
    Read an image and extract key points from it
    :param img: image to extract, type is ndarray
           dialation: bounding box dialtion rate
    :return: A list of person info. person=>[head,body] posision

    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # img = cv2.imread(img)
    pts = []  # the key points

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    rects = detector(img_gray, 0)
    # print(rects)
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y]
                               for p in predictor(img, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = [point[0, 0], point[0, 1]]
            pts.append(pos)

    # head bounding boxes
    height,width = img_gray.shape
    # print(width,height)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    head_bounding_boxes = []
    for subject in rects:
        h = subject.height()
        w = subject.width()
        top = subject.top()
        bottom = subject.bottom()
        left = subject.left()
        right = subject.right()
        x=int(top-dialation*h) if int(top-dialation*h)>=0 else 0
        y=int(left-dialation*w) if int(top-dialation*h)>=0 else 0
        h=int(h+2*dialation*h) if int(top-dialation*h)+int(h+2*dialation*h)<height else height-int(h+2*dialation*h)
        w=int(w+2*dialation*w) if int(top-dialation*w)+int(w+2*dialation*w)<width else width-int(w+2*dialation*w)
        head_bounding_boxes.append(
            [x, y, h,w])
    print(head_bounding_boxes)

    haar_upper_body_cascade = cv2.CascadeClassifier(
        "./haarcascade_upperbody.xml")
    upper_body_bounding_boxes = haar_upper_body_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(25, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(upper_body_bounding_boxes)<len(head_bounding_boxes):
        upper_body_bounding_boxes=[]
        for subject in head_bounding_boxes:
            left = subject[1]-0.25*subject[3]
            top = subject[0]+subject[2]
            w = subject[2]*2
            h = height - top
            upper_body_bounding_boxes.append([int(left),int(top),int(h),int(w)])
    print(upper_body_bounding_boxes)

    data = [head_bounding_boxes, upper_body_bounding_boxes]
    data = np.array(data)
    data = np.swapaxes(data, 0, 1)

    # return head_bounding_boxes, upper_body_bounding_boxes
    return data


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # cv2读取图像
    img = cv2.imread("imgs/homo_test_1/photo1.jpg")
    extract(img)
    # 取灰度
    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 人脸数rects
    # rects = detector(img_gray, 0)
    # for i in range(len(rects)):
    #     landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    #     for idx, point in enumerate(landmarks):
    # 68点的坐标
    # pos = (point[0, 0], point[0, 1])
    # print(idx, pos)

    # 利用cv2.circle给每个特征点画一个圈，共68个
    # cv2.circle(img, pos, 3, color=(0, 255, 0))
    # 利用cv2.putText输出1-68
    # font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
