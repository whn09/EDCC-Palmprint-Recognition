# coding: utf-8
import os
import cv2
import math
import time
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


def balance_color(filename):
    im = Image.open(filename)
    tb = im.histogram()  # 获得im的直方图

    totalpixel = 0  # 用于统计像素总数，即MN
    maptb = []  # 存储映射关系
    count = len(tb)
    for i in range(count):
        totalpixel += tb[i]
        maptb.append(totalpixel)

    for i in range(count):
        maptb[i] = int(round((maptb[i] * (count - 1)) / totalpixel))

    def histogram(light):
        return maptb[light]

    out = im.point(histogram)  # 对im应用直方图均衡
    return np.asarray(out)


def get_contours(filename, left=True, debug=False):
    # 读取原图并转化为灰度图
    # img = cv2.imread(filename)
    # cv2.imshow('img', img)
    # print('img.shape:', img.shape)
    # imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 读取原图并转化为灰度图（利用直方图均衡算法）
    imggray = balance_color(filename)
    if debug:
        # cv2.imshow('imggray', imggray)
        cv2.imwrite(filename + '.gray.png', imggray)

    # 将灰度图转为二值图，只保留特别白的部分
    binary_thres = 225
    ret, imgbinary = cv2.threshold(imggray, binary_thres, 255, 0)
    if debug:
        # cv2.imshow('imgbinary', imgbinary)
        cv2.imwrite(filename + '.binary.png', imgbinary)

    # 将二值图进行膨胀和腐蚀
    # kernel = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #    imgdilate = cv2.dilate(imgbinary, kernel, iterations=1)
    #    cv2.imshow('imgdilate', imgdilate)
    #    imgerosion = cv2.erode(imgdilate, kernel, iterations=1)
    #    cv2.imshow('imgerosion', imgerosion)
    #    imgopening = cv2.morphologyEx(imgbinary, cv2.MORPH_OPEN, kernel)
    #    cv2.imshow('imgopening', imgopening)
    imgclosing = cv2.morphologyEx(imgbinary, cv2.MORPH_CLOSE, kernel)  # iterations=2
    if debug:
        # cv2.imshow('imgclosing', imgclosing)
        cv2.imwrite(filename + '.closing.png', imgclosing)

    # 将膨胀和腐蚀之后的二值图进行模糊化
    for i in range(2):
        imgclosing = cv2.medianBlur(imgclosing, 5)
    imgmedian = imgclosing
    # imgmedian = cv2.medianBlur(imgclosing, 3)
    # imgmedian = cv2.GaussianBlur(imgclosing, (3, 3), 1)
    # imgmedian = cv2.blur(imgclosing, (3, 3))
    if debug:
        # cv2.imshow('imgmedian', imgmedian)
        cv2.imwrite(filename + '.median.png', imgmedian)

    # 在模糊化之后的二值图中找到所有的轮廓（cnts）
    imgcontours, cnts, hierarchy = cv2.findContours(imgmedian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('cnts:', np.size(cnts))  # **得到该图中总的轮廓数量**
    # print(cnts[0])  # 打印出第一个轮廓的所有点的坐标， 更改此处的0，为0--（总轮廓数-1），可打印出相应轮廓所有点的坐标
    # print(hierarchy)  # **打印出相应轮廓之间的关系**

    # 将轮廓按面积从大到小排序
    cnts_list = []
    for i in range(np.size(cnts)):
        # print(i, len(cnts[i]))
        # cnts_list.append((i, len(cnts[i])))  # 轮廓点数目
        cnts_list.append((i, cv2.contourArea(cnts[i])))  # 轮廓面积（好于轮廓点）
    cnts_list.sort(key=lambda x: -x[1])
    # print('cnts_list:', cnts_list)

    # 过滤面积过小（contour_area_thres）和大多数点（noise_thres）位于左上右下边缘的轮廓
    height, width = imgcontours.shape
    if left:
        left_thres, up_thres, right_thres, down_thres = 0.0, 0.0, 1.0, 1.0  # default: 0.15, 0.15, 0.75, 0.85
    else:
        left_thres, up_thres, right_thres, down_thres = 0.0, 0.0, 1.0, 1.0  # default: 0.2, 0.15, 0.9, 0.85
    noise_thres = 0.5
    contour_area_thres = 10  # default is 20
    max_palm_cnt = 30  # default is 20
    palm_cnt = 0
    palmprints = []
    if debug:
        cv2.line(imgcontours, (int(width * left_thres), 0), (int(width * left_thres), height), (255, 0, 0, 255),
                 1)  # green
        cv2.line(imgcontours, (int(width * right_thres), 0), (int(width * right_thres), height), (255, 0, 0, 255),
                 1)  # blue
        cv2.line(imgcontours, (0, int(height * up_thres)), (width, int(height * up_thres)), (255, 0, 0, 255), 1)
        cv2.line(imgcontours, (0, int(height * down_thres)), (width, int(height * down_thres)), (255, 0, 0, 255), 1)
        # cv2.imshow('imgcontours', imgcontours)  # **显示返回值image，其实与输入参数的imgbinary原图没啥区别**
        cv2.imwrite(filename + '.contours.png', imgcontours)
    for i in range(len(cnts_list)):
        if cnts_list[i][1] >= contour_area_thres:
            contour = cnts[cnts_list[i][0]]
            leftcnt, upcnt, rightcnt, downcnt = 0, 0, 0, 0
            for cont in contour:
                x = cont[0][0]
                y = cont[0][1]
                if x < left_thres * width:
                    leftcnt += 1
                if x > right_thres * width:
                    rightcnt += 1
                if y < up_thres * height:
                    upcnt += 1
                if y > down_thres * height:
                    downcnt += 1
            # print(i, len(contour), leftcnt, upcnt, rightcnt, downcnt, noise_thres * len(contour))
            if leftcnt > noise_thres * len(contour) or \
                    rightcnt > noise_thres * len(contour) or \
                    upcnt > noise_thres * len(contour) or \
                    downcnt > noise_thres * len(contour):
                continue
            else:
                if palm_cnt < max_palm_cnt:
                    palmprints.append(contour)
                    palm_cnt += 1
                else:
                    break

    # 找到每个轮廓的：最左上方和右下方的点（左手），最右上方和左下方的点（右手），并连成线
    lines = []
    for contour in palmprints:
        hull = cv2.convexHull(contour)
        rect = cv2.boundingRect(hull)
        if left:
            lines.append([(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])])
        else:
            lines.append([(rect[0] + rect[2], rect[1]), (rect[0], rect[1] + rect[3])])

    # 将所有的线连起来：连接一个线段的终点到最近的线段的起点，且夹角不能太大或距离不能太远或距离特别近
    end_starts = {}
    start_ends = {}
    min_dis_thres = 20  # default is 20
    max_dis_thres = 60  # default is 60
    left_angle_x_thres_min = -1.0 / 6.0
    left_angle_x_thres_max = 1.0  # 2.0 / 3.0
    right_angle_x_thres_min = -1.0  # -2.0 / 3.0
    right_angle_x_thres_max = 1.0 / 6.0
    for i in range(len(lines)):
        end = lines[i][1]
        min_dis = -1
        min_start = None
        min_angle_x = 0
        min_angle_y = 0
        min_j = -1
        for j in range(len(lines)):
            if i != j:
                start = lines[j][0]
                dis = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                if dis > 0:
                    angle_x = (start[0] - end[0]) / dis
                    angle_y = (start[1] - end[1]) / dis
                    if min_dis == -1 or dis < min_dis:
                        min_dis = dis
                        min_start = start
                        min_angle_x = angle_x
                        min_angle_y = angle_y
                        min_j = j
        # print('i:', i, 'min_j:', min_j, 'min_dis:', min_dis, 'min_angle_x:', min_angle_x, 'min_angle_y:', min_angle_y)
        if min_start is not None and \
                min_dis <= max_dis_thres and \
                (min_angle_y >= 0 or min_dis <= min_dis_thres) and \
                ((left and min_angle_x >= left_angle_x_thres_min and min_angle_x <= left_angle_x_thres_max) or \
                 (not left and min_angle_x >= right_angle_x_thres_min and min_angle_x <= right_angle_x_thres_max) or \
                 min_dis <= min_dis_thres):
            end_starts[i] = min_j
            start_ends[min_j] = i

    # 找到新连接出来的线段的头和尾（目前只用头即可），如果是独立的一条线，则只保留较长的线段
    heads = []
    tails = []
    length_thres = 60
    for i in range(len(lines)):
        length = math.sqrt((lines[i][0][0] - lines[i][1][0]) ** 2 + (lines[i][0][1] - lines[i][1][1]) ** 2)
        if (i in end_starts or length >= length_thres) and i not in start_ends:
            heads.append(i)
        if i not in end_starts and (i in start_ends or length >= length_thres):
            tails.append(i)

    # 将新线段按先后顺序连起来
    new_lines = []
    for head in heads:
        line = [head]
        while head in end_starts and end_starts[head] not in line:
            line.append(end_starts[head])
            head = end_starts[head]
        new_lines.append(line)

    # 将所有新线段画到一个透明图片上，并做模糊处理，并保存
    alpha_shape = (imggray.shape[0], imggray.shape[1], 4)
    imgpalmprint = np.zeros(alpha_shape, np.uint8)
    palmprint_color = (0, 0, 255, 255)  # red
    # palmprint_color = (0, 255, 0, 255)  # green
    contour_area_max_thres = 200
    for line in new_lines:
        for i in range(len(line)):
            contour = palmprints[line[i]]
            contour_area = cv2.contourArea(contour)
            if contour_area >= contour_area_max_thres:  # 如果原来的轮廓面积比较大，则直接画这个轮廓，否则画直线
                cv2.drawContours(imgpalmprint, palmprints, line[i], palmprint_color, 6)
            else:
                cv2.line(imgpalmprint, lines[line[i]][0], lines[line[i]][1], palmprint_color, 6)
            if i < len(line) - 1:
                cv2.line(imgpalmprint, lines[line[i]][1], lines[line[i + 1]][0], palmprint_color, 6)
    imgpalmprint = cv2.GaussianBlur(imgpalmprint, (3, 3), 1)
    # imgpalmprint = cv2.blur(imgpalmprint, (3, 3))
    # cv2.imshow('imgpalmprint', imgpalmprint)
    cv2.imwrite(filename + '.palmprint.png', imgpalmprint)

    # 将掌纹图片和原始图片叠加
    if debug:
        imgorigin = cv2.imread(filename[:-8])
        if imgorigin is None:
            imgorigin = cv2.imread(filename[:-9])
        imgpalmprint = cv2.resize(imgpalmprint, (imgorigin.shape[1], imgorigin.shape[0]))
        b = imgorigin[:, :, 0] * 1.0 + imgpalmprint[:, :, 0] * 1.0
        g = imgorigin[:, :, 1] * 1.0 + imgpalmprint[:, :, 1] * 1.0
        r = imgorigin[:, :, 2] * 1.0 + imgpalmprint[:, :, 2] * 1.0
        imgmerge = cv2.merge([b, g, r])
        cv2.imwrite(filename + '.merge.png', imgmerge)

    return filename + '.palmprint.png'


def enhance_image(filename):
    out_filename = filename + '.out2.png'
    img = cv2.imread(filename, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    img = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=5, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)  # TODO 放大或缩小成500*500图片，缩小需用INTER_AREA
    cv2.imwrite(out_filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return out_filename


def GetGaborKernelReal(width, height, dimension, direction, kmax, f, sigma, ktype=cv2.CV_64F):
    half_width = width / 2
    half_height = height / 2
    Qu = np.pi * direction / 10  # 10 is directions
    sqsigma = sigma * sigma
    Kv = kmax / math.pow(f, dimension)
    postmean = math.exp(-sqsigma / 2)
    kernel = np.zeros((width, height), dtype=np.float64)
    # print('half_width:', half_width)
    # print('half_height:', half_height)
    # print('Qu:', Qu)
    # print('sqsigma:', sqsigma)
    # print('Kv:', Kv)
    # print('postmean:', postmean)
    # print('kernel:', kernel)
    for i in range(-half_height, half_height + 1):
        for j in range(-half_width, half_width + 1):
            tmp1 = math.exp(-(Kv * Kv * (i * i + j * j)) / (2 * sqsigma))
            tmp2 = math.cos(Kv * math.cos(Qu) * j + Kv * math.sin(Qu) * i) - postmean
            kernel[i + half_height][j + half_width] = float(Kv * Kv * tmp1 * tmp2 / sqsigma)
    # print('kernel:', kernel)
    return kernel


# 构建Gabor滤波器
def build_filters():
    filters = []
    ksize = 5  # gabor尺度
    sigma = 2.0 * np.pi
    lambd = np.pi * 0.5  # 波长
    gamma = 2.0  # TODO
    psi = np.pi * 0.5  # TODO
    # method1: opencv自带函数
    # for theta in np.arange(0, np.pi, np.pi / 10):  # gabor方向
    #    kern = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma, theta=theta, lambd=lambd,
    #                              gamma=gamma, psi=psi, ktype=cv2.CV_64F)
    #    #print('kern:', kern)
    #    filters.append(kern)
    # method2: EDCC作者实现函数
    for i in range(10):  # 10 is directions
        kern = GetGaborKernelReal(ksize, ksize, 0, i, lambd, math.sqrt(2.0), sigma, cv2.CV_64F)
        filters.append(kern)
    return filters


# Gabor滤波过程
def process(img, kern):
    accum = np.zeros_like(img, dtype=np.float64)
    fimg = cv2.filter2D(img, cv2.CV_64F, kern)
    accum = cv2.normalize(fimg, accum, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return accum


# Gabor特征提取
def getGabor(filename, filters, debug=False):
    img = cv2.imread(filename)
    res = []
    for i in range(len(filters)):
        res1 = process(img, filters[i])
        res.append(res1)
    if debug:
        for temp in range(len(res)):
            temp_image = np.array(res[temp] * 255, dtype=np.uint8)
            cv2.imwrite(filename + '.channel' + str(temp) + '.png', temp_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return res


def GenEDCCoding(gabor_filter_result, width, height, directions):
    c_ = np.zeros((width, height), np.uint8)
    cs_ = np.zeros((width, height), np.uint8)
    for h in range(height):
        for w in range(width):
            max_response = -999999  # -DBL_MAX
            max_direction = -1
            for d in range(directions):
                response = gabor_filter_result[d][h][w][0]  # TODO gabor_filter_result[d][h][w] is array[3]
                if response > max_response:
                    max_response = response
                    max_direction = d
            c_[h][w] = max_direction
            if max_direction == directions - 1:
                c_left = 0
            else:
                c_left = max_direction + 1
            if max_direction == 0:
                c_right = directions - 1
            else:
                c_right = max_direction - 1
            cs_[h][w] = gabor_filter_result[c_left][h][w][0] >= gabor_filter_result[c_right][h][w][0] and 1 or 0
    return c_, cs_


def GenCodingBytesFastMode(c_, cs_):
    data = []
    for h in range(len(c_)):
        for w in range(len(c_[0])):
            coding_c = c_[h][w]
            coding_cs = cs_[h][w]
            da = 0
            da |= coding_c << 4
            da |= coding_cs
            data.append(da)
    return data


def ExecuteMatchingWhenFastCodingMode(coding1, coding2):
    match_score = 0
    for i in range(len(coding1)):
        c1 = coding1[i]
        c2 = coding2[i]
        cmp_value = c1 ^ c2
        if cmp_value == 0x00:
            match_score += 2
        elif cmp_value < 0x10:
            match_score += 1
        else:
            match_score += 0
    score = match_score / (2.0 * len(coding1))
    return score


def get_encoding(filename, filters, debug=False):
    start = time.time()
    res = getGabor(filename, filters, debug)
    end = time.time()
    #print('getGabor time:', end-start)
    c_, cs_ = GenEDCCoding(res, 500, 500, 10)  # TODO
    #print('c_:', len(c_))
    #print('cs_:', len(cs_))
    end2 = time.time()
    #print('GenEDCCoding time:', end2 - end)
    data = GenCodingBytesFastMode(c_, cs_)
    #print('data:', len(data))
    end3 = time.time()
    #print('GenCodingBytesFastMode time:', end3 - end2)
    if debug:
        data_image = np.reshape(data, (500, 500))  # TODO
        cv2.imwrite(filename + '.data_image.png', data_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return data


def evaluate_encodings(dir):
    filenames = sorted(os.listdir(dir))
    cnt = 0
    encoding_filenames = []
    for filename in filenames:
        if filename.endswith('.data_image.png'):  # and cnt < 60:
            cnt += 1
            #print(filename)
            filename = os.path.join(dir, filename)
            # TODO evaluate encoding
            encoding_filenames.append(filename)
    print('encoding_filenames:', len(encoding_filenames))
    y_test = []
    y_score = []
    for i in range(len(encoding_filenames)):
        print('evaluate_encodings:', i, '/', len(encoding_filenames))
        start = time.time()
        encoding1 = cv2.imread(encoding_filenames[i], cv2.IMREAD_GRAYSCALE)
        encoding1 = np.reshape(encoding1, (encoding1.shape[0]*encoding1.shape[1], 1))
        encoding1_str = ''
        for code in encoding1:
            encoding1_str += chr(code)
        encoding1 = encoding1_str
        end = time.time()
        print(i, 'encoding1 time:', end - start)
        for j in range(i, len(encoding_filenames)):
            if i/10 == j/10:
                y_test.append(1)
            else:
                y_test.append(0)
            encoding2 = cv2.imread(encoding_filenames[j], cv2.IMREAD_GRAYSCALE)
            encoding2 = np.reshape(encoding2, (encoding2.shape[0] * encoding2.shape[1], 1))
            encoding2_str = ''
            for code in encoding2:
                encoding2_str += chr(code)
            encoding2 = encoding2_str
            start1 = time.time()
            #score = ExecuteMatchingWhenFastCodingMode(encoding1, encoding2)
            score = edcc_api.GetTwoPalmprintCodingMatchScore(encoding1, encoding2)
            end1 = time.time()
            print(i, j, 'match time:', end1 - start1)
            y_score.append(score)
        end = time.time()
        print(i, 'time:', end-start)
    from DeepRecommender.graph_utils import roc, ks, f1score
    print('***********evaluate_encodings***********')
    f1score(y_test, y_score)
    roc(y_test, y_score, 'evaluate_encodings_roc.png')
    ks(y_test, y_score, 'evaluate_encodings_ks.png')


def evaluate_encodings_cpp(dir):
    from edcc_adapter import *
    edcc_api = EdccApi()
    configPath = os.path.normpath(os.path.join(os.getcwd(), "edcc_config/config.json"))

    filenames = sorted(os.listdir(dir))
    cnt = 0
    encoding_filenames = []
    for filename in filenames:
        if filename.endswith('.bmp'):  # and cnt < 60:
            cnt += 1
            #print(filename)
            filename = os.path.join(dir, filename)
            # TODO evaluate encoding
            encoding_filenames.append(filename)
    print('encoding_filenames:', len(encoding_filenames))
    ids = []
    y_test = []
    y_score = []
    for i in range(len(encoding_filenames)):
        print('evaluate_encodings:', i, '/', len(encoding_filenames))
        start = time.time()
        encoding1, codingLen1 = edcc_api.GetEDCCCoding(encoding_filenames[i], configPath)
        end = time.time()
        #print(i, 'encoding1 time:', end - start)
        for j in range(i, len(encoding_filenames)):
            if i/10 == j/10:
                y_test.append(1)
            else:
                y_test.append(0)
            encoding2, codingLen2 = edcc_api.GetEDCCCoding(encoding_filenames[j], configPath)
            start1 = time.time()
            score = edcc_api.GetTwoPalmprintCodingMatchScore(encoding1, encoding2)
            end1 = time.time()
            #print(i, j, 'match time:', end1 - start1)
            y_score.append(score)
            ids.append(str(i)+','+str(j))
        end = time.time()
        print(i, 'time:', end-start)
    from DeepRecommender.graph_utils import roc, ks, f1score, dump_score
    print('***********evaluate_encodings***********')
    f1score(y_test, y_score)
    roc(y_test, y_score, 'evaluate_encodings_roc.png')
    ks(y_test, y_score, 'evaluate_encodings_ks.png')
    dump_score(ids, y_test, y_score, 'evaluate_encodings_preds.csv')


def draw_pic(n_clusters, core_samples_mask, labels, X):
    import matplotlib.pyplot as plt

    ''' 开始绘制图片 '''
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        #print('class_member_mask & core_samples_mask:', class_member_mask & core_samples_mask)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=4)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=2)

    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.show()


def encoding_clusters_tmp():
    fin = open('evaluate_encodings_preds.csv', 'r')
    lines = fin.readlines()
    max_id = int(lines[-1].strip().split(',')[0])

    # from sklearn.datasets.samples_generator import make_blobs
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics.pairwise import euclidean_distances
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, labels_true = make_blobs(n_samples=750, centers=centers,
    #                            cluster_std=0.4, random_state=0)
    # X = StandardScaler().fit_transform(X)
    # distance_matrix = euclidean_distances(X)
    # print('X:', X.shape)
    # print('distance_matrix:', distance_matrix.shape)

    X = [[i, j] for j in range(max_id+1) for i in range(max_id+1)]
    X = np.array(X)
    print('X:', X.shape)
    distance_matrix = [[99 for _ in range(X.shape[0])] for _ in range(X.shape[0])]

    for line in lines:
        params = line.strip().split(',')
        distance_matrix[int(params[0])][int(params[1])] = 1.0-float(params[2])
        distance_matrix[int(params[1])][int(params[0])] = 1.0-float(params[2])
    distance_matrix = np.array(distance_matrix)
    print('distance_matrix:', distance_matrix.shape)

    db1 = DBSCAN(eps=0.01, min_samples=5, metric='precomputed').fit(distance_matrix)
    labels1 = db1.labels_  # 每个点的标签
    core_samples_mask1 = np.zeros_like(db1.labels_, dtype=bool)
    core_samples_mask1[db1.core_sample_indices_] = True
    n_clusters1 = len(set(labels1)) - (1 if -1 in labels1 else 0)  # 类的数目
    print('n_clusters1:', n_clusters1)
    print('core_samples_mask1:', len(core_samples_mask1))
    print('labels1:', len(labels1))
    print('X:', len(X))
    draw_pic(n_clusters1, core_samples_mask1, labels1, X)
    fin.close()


def draw_matrix(X, distance_matrix):
    import matplotlib.pyplot as plt

    ''' 开始绘制图片 '''
    # Black removed and is used for noise instead.
    colors = plt.cm.Spectral(np.linspace(0, 1, 2))
    for x in X:
        color = 'k'
        if distance_matrix[x[0]][x[1]] < 0.91:
            color = 'r'
        plt.plot(x[0], x[1], 'o', markerfacecolor=color,
                 markeredgecolor=color, markersize=2)

    plt.title('Matrix')
    plt.show()


def encoding_clusters():
    fin = open('evaluate_encodings_preds.csv', 'r')
    lines = fin.readlines()
    max_id = int(lines[-1].strip().split(',')[0])

    X = [[i, j] for j in range(max_id+1) for i in range(max_id+1)]
    X = np.array(X)
    print('X:', X.shape)
    distance_matrix = [[99 for _ in range(max_id+1)] for _ in range(max_id+1)]

    for line in lines:
        params = line.strip().split(',')
        distance_matrix[int(params[0])][int(params[1])] = 1.0-float(params[2])
        distance_matrix[int(params[1])][int(params[0])] = 1.0-float(params[2])
    distance_matrix = np.array(distance_matrix)
    print('distance_matrix:', distance_matrix.shape)
    print('distance_matrix:', distance_matrix)
    draw_matrix(X, distance_matrix)


if __name__ == '__main__':

    filters = build_filters()
    for i in range(len(filters)):
        filter_image = np.array(filters[i] * 255, dtype=np.uint8)
        cv2.imwrite('filter_' + str(i) + '.png', filter_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    #    filename = 'test1.png'
    #    get_contours(filename)
    #    exit(-1)
    #dir = '../../../../../dataset/PalmPrint/Tongji/ROI/session1/'
    dir = '../../database/Tongji/ROI/session1/'
    #dir = '../../database/Tongji/ROI/session2/'
    # dir = './left/'
    # dir = './right/'
    # dir = './palm_test_left/'
    # dir = './palm_test_right/'
    #dir = './right_20180830/'

    #evaluate_encodings(dir)
    #evaluate_encodings_cpp(dir)
    encoding_clusters()
    exit(-1)

    filenames = sorted(os.listdir(dir))
    cnt = 0
    encodings = []
    for filename in filenames:
        cnt += 1
        if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.bmp'):
            print(filename)
            start = time.time()
            filename = os.path.join(dir, filename)
            filename = enhance_image(filename)
            encoding = get_encoding(filename, filters, debug=False)
#            encodings.append(encoding)
#            #get_contours(filename, left=False, debug=False)
#            get_contours(filename, left=True, debug=False)
            end = time.time()
            print('time:', end-start)

        # if filename.endswith('.out.png'):
        #    print(filename)
        #    filename = os.path.join(dir, filename)
        #    get_contours(filename, left=False, debug=True)
#    for i in range(len(encodings)-1):
#        start = time.time()
#        score = ExecuteMatchingWhenFastCodingMode(encodings[i], encodings[i+1])
#        end = time.time()
#        print(str(i)+' '+str(i+1)+':', score, 'time:', end-start)

