import numpy as np
import cv2
import dlib


def extract(img, dialation=[0.5,0.8,0.5,0.05]):
    '''
    Read an image and extract key points from it
    :param img: image to extract, type is ndarray
           dialation: bounding box dialtion rate left,up,right,down
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
        x=int(left-dialation[0]*w) if int(top-dialation[0]*h)>=0 else 0
        y=int(top-dialation[1]*h) if int(top-dialation[1]*h)>=0 else 0
        w=int(w+(dialation[0]+dialation[2])*w) if x+int(w+(dialation[0]+dialation[2])*w)<width else width-x
        h=int(h+(dialation[1]+dialation[3])*h) if y+int(h+(dialation[1]+dialation[3])*h)<height else height-y
        head_bounding_boxes.append(
            [x, y, w,h])
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
            left = subject[0]-0.25*subject[2]
            top = subject[1]+subject[3]
            w = subject[2]*1.5 if left+w<width-1 else width
            h = height - top
            upper_body_bounding_boxes.append([int(left),int(top),int(w),int(h)])
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
    subject_info = extract(img)
    for idx,subject in enumerate(subject_info):
        head_bb = subject[0]
        body_bb = subject[1]
        cv2.rectangle(img, (head_bb[0],head_bb[1]), (head_bb[0]+head_bb[2], head_bb[1]+head_bb[3]), (0, 255, 0), 2)
        cv2.rectangle(img, (body_bb[0],body_bb[1]), (body_bb[0]+body_bb[2], body_bb[1]+body_bb[3]), (0, 255, 0), 2)
        print(head_bb,body_bb)
    print(subject_info.shape)
    # cv2.circle(img, pos, 3, color=(0, 255, 0))
    cv2.imshow("img", img)
    cv2.waitKey(0)
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
