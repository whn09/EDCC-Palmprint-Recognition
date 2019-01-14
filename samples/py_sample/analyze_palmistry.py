import os
import cv2
import sys
import json
import numpy as np

sys.path.append('../../include/')
from edcc_adapter import *

sys.path.append('../../../DeepRecommender/')
from graph_utils import *

DIR = '../../../../../dataset/'
IMAGE_DIR = DIR + 'palmistry/image/'
CODING_DIR = DIR + 'palmistry/coding/'
DATA_DIR = DIR + 'palmistry/data/'
DUMP_FILENAME = DIR + 'palmistry/images_and_codings.csv'
BASE_URL = 'http://52.70.8.5:8888/tree/workspace/dataset/palmistry/'
CONFIG_PATH = os.path.normpath(os.path.join(os.getcwd(), 'edcc_config/config.json'))


def get_images(dir):
    dirs = os.listdir(dir)
    images = {}  # muid:[[origin_image_path, analysis_image_path, create_time]]
    for di in dirs:
        params = di.split('-')
        muid = '-'.join(params[:-1])
        create_time = params[-1]
        if muid not in images:
            images[muid] = []
        images[muid].append([di + '/origin.jpg', di + '/analysis.png', create_time])
    print('dirs:', len(dirs))
    print('images:', len(images))
    return images


def get_codings(dir):
    dirs = os.listdir(dir)
    codings = {}  # muid:[[origin_coding_path, create_time]]
    for di in dirs:
        params = di.split('-')
        muid = '-'.join(params[:-1])
        create_time = params[-1]
        if muid not in codings:
            codings[muid] = []
        codings[muid].append([di + '/origin_coding', create_time])
    print('dirs:', len(dirs))
    print('codings:', len(codings))
    return codings


def merge_images_and_codings(images, codings):
    images_and_codings = {}  # muid:[[origin_image_path, analysis_image_path, origin_coding_path, create_time]]
    images_and_codings_cnt = 0
    for muid, values in images.items():
        if muid in codings:
            for value in values:
                for coding in codings[muid]:
                    if value[2] == coding[1]:
                        if muid not in images_and_codings:
                            images_and_codings[muid] = []
                        images_and_codings[muid].append([value[0], value[1], coding[0], coding[1]])
                        images_and_codings_cnt += 1
                        break
    print('images_and_codings_cnt:', images_and_codings_cnt)
    print('images_and_codings:', len(images_and_codings))
    return images_and_codings


def dump_images_and_codings(images_and_codings, filename, base_url):
    fout = open(filename, 'w')
    fout.write('muid,origin_image_path,analysis_image_path,origin_coding_path,create_time\n')
    for muid, values in images_and_codings.items():
        for value in values:
            fout.write(muid + ',' + base_url + 'image/' + value[0] + ',' + base_url + 'image/' + value[
                1] + ',' + base_url + 'coding/' + value[2] + ',' + value[3] + '\n')
    fout.close()


def evaluate_match(images_and_codings, dir):
    pre_predictPalmprintCode = ''
    y_test = []
    y_score = []
    for muid, values in images_and_codings.items():
        for i, value in enumerate(values):
            predictPalmprintCode, codingLen = edcc_api.GetEDCCCoding(
                dir + value[0], CONFIG_PATH)
            if pre_predictPalmprintCode != '':
                matchScore = edcc_api.GetTwoPalmprintCodingMatchScore(
                    predictPalmprintCode, pre_predictPalmprintCode)
                if i == 0:
                    y_test.append(0)
                else:
                    y_test.append(1)
                y_score.append(matchScore)
            pre_predictPalmprintCode = predictPalmprintCode
    roc(y_test, y_score, dir + 'palmistry/roc.png')


def evaluate_match_new(infos, dir):
    pre_predictPalmprintCode = ''
    y_test = []
    y_score = []
    for muid, values in infos.items():
        print('evaluate_match_new:', muid, len(values))
        for i, value in enumerate(values):
            img = cv2.imread(dir + value[0])
            # print('img.shape[0]', img.shape[0])
            # print('value[4]:', value[4])
            # print('value[6]:', value[6])
            # print('img.shape[1]', img.shape[1])
            # print('value[3]:', value[3])
            # print('value[5]:', value[5])
            crop_filename = dir + value[0][:-4] + '_crop.png'
            if not os.path.exists(crop_filename):
                crop_img = img[int(img.shape[0] * value[4]):int(img.shape[0] * value[4] + img.shape[0] * value[6]),
                           int(img.shape[1] * value[3]):int(img.shape[1] * value[3] + img.shape[1] * value[5])]
                cv2.imwrite(crop_filename, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #print('crop_filename:', crop_filename)
            #print('CONFIG_PATH:', CONFIG_PATH)
            predictPalmprintCode, codingLen = edcc_api.GetEDCCCoding(
                crop_filename, CONFIG_PATH)
            if pre_predictPalmprintCode != '':
                matchScore = edcc_api.GetTwoPalmprintCodingMatchScore(
                    predictPalmprintCode, pre_predictPalmprintCode)
                if i == 0:
                    y_test.append(0)
                else:
                    y_test.append(1)
                y_score.append(matchScore)
            pre_predictPalmprintCode = predictPalmprintCode
    roc(y_test, y_score, dir + 'palmistry/roc.png')
    ks(y_test, y_score, dir + 'palmistry/ks.png')
    f1score(y_test, y_score, step=0.001, base=0.08)


def evaluate_search_new(infos, dir):
    all_codings = []  # [(muid, coding)]
    hit = 0
    not_hit = 0
    should_hit = 0
    for muid, values in infos.items():
        #print('get all_codings:', muid, len(values))
        if len(values) > 1:
            should_hit += len(values)
        for i, value in enumerate(values):
            img = cv2.imread(dir + value[0])
            # print('img.shape[0]', img.shape[0])
            # print('value[4]:', value[4])
            # print('value[6]:', value[6])
            # print('img.shape[1]', img.shape[1])
            # print('value[3]:', value[3])
            # print('value[5]:', value[5])
            crop_filename = dir + value[0][:-4] + '_crop.png'
            if not os.path.exists(crop_filename):
                crop_img = img[int(img.shape[0] * value[4]):int(img.shape[0] * value[4] + img.shape[0] * value[6]),
                           int(img.shape[1] * value[3]):int(img.shape[1] * value[3] + img.shape[1] * value[5])]
                cv2.imwrite(crop_filename, crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #print('crop_filename:', crop_filename)
            #print('CONFIG_PATH:', CONFIG_PATH)
            predictPalmprintCode, codingLen = edcc_api.GetEDCCCoding(
                crop_filename, CONFIG_PATH)
            all_codings.append((muid, predictPalmprintCode))
    print('should_hit:', should_hit)
    print('all_codings:', len(all_codings))
    for i in range(len(all_codings)):
        if i % 100 == 0:
            print('evaluate_search_new i:', i, '/', len(all_codings))
        muidi = all_codings[i][0]
        codingi = all_codings[i][1]
        max_matchScore = -1
        max_muid = ''
        for j in range(len(all_codings)):
            if i != j:
                muidj = all_codings[j][0]
                codingj = all_codings[j][1]
                matchScore = edcc_api.GetTwoPalmprintCodingMatchScore(codingi, codingj)
                if matchScore > max_matchScore:
                    max_matchScore = matchScore
                    max_muid = muidj
        if max_muid != '' and muidi == max_muid and max_matchScore >= 0.08:
            hit += 1
        else:
            not_hit += 1
    print('hit:', hit, float(hit)/len(all_codings))
    print('not_hit:', not_hit, float(not_hit)/len(all_codings))


def get_infos(info_dir, dir):
    infos = {}  # muid:[[origin_image_s3_key, image_s3_key, image_coding_s3_key, pointx_per, pointy_per, width_per, height_per, hand, create_time]]
    infos_cnt = 0
    all_lines = 0
    all_count = 0
    all_files = 0
    for filename in os.listdir(info_dir):
        if not filename.endswith('json'):
            continue
        fin = open(os.path.join(info_dir, filename), 'r')
        try:
            js = json.load(fin)
        except:
            print('ERROR!', filename)
            continue
        lines = js['Items']
        all_lines += len(lines)
        all_count += js['Count']
        all_files += 1
        for line in lines:
            params = line
            if len(params) == 15:
                if 'muid' in params and 'S' in params['muid']:
                    muid = str(params['muid']['S'])
                else:
                    muid = ''
                if 'ick' in params and 'S' in params['ick']:
                    image_coding_s3_key = str(params['ick']['S'])
                else:
                    image_coding_s3_key = ''
                if 'oik' in params and 'S' in params['oik']:
                    origin_image_s3_key = str(params['oik']['S'])
                else:
                    origin_image_s3_key = ''
                if 'ik' in params and 'S' in params['ik']:
                    image_s3_key = str(params['ik']['S'])
                else:
                    image_s3_key = ''
                if 'ct' in params and 'N' in params['ct']:
                    create_time = int(params['ct']['N'])
                else:
                    create_time = ''
                if 'xp' in params and 'N' in params['xp']:
                    pointx_per = float(params['xp']['N'])
                else:
                    pointx_per = 0
                if 'yp' in params and 'N' in params['yp']:
                    pointy_per = float(params['yp']['N'])
                else:
                    pointy_per = 0
                if 'wp' in params and 'N' in params['wp']:
                    width_per = float(params['wp']['N'])
                else:
                    width_per = 0
                if 'hp' in params and 'N' in params['hp']:
                    height_per = float(params['hp']['N'])
                else:
                    height_per = 0
                if 'hand' in params and 'N' in params['hand']:  # 0-left, 1-right
                    hand = int(params['hand']['N'])
                else:
                    hand = 0

                if os.path.exists(dir + origin_image_s3_key):  # TODO only use local images
                    if muid not in infos:
                        infos[muid] = []
                    infos[muid].append(
                        [origin_image_s3_key, image_s3_key, image_coding_s3_key, pointx_per, pointy_per, width_per,
                         height_per, hand, create_time])
                    infos_cnt += 1

    print('all_lines:', all_lines)
    print('all_count:', all_count)
    print('all_files:', all_files)
    print('infos:', len(infos))
    print('infos_cnt:', infos_cnt)
    return infos


def analyze_coding(filename):
    print('filename:', filename)
    fin = open(filename, 'r')
    # TODO infer (may be wrong)
    line1 = fin.readline().strip()
    for i in range(len(line1)):
        print('line1:', ord(line1[i]))
    imageSizeW = ord(line1[0])
    imageSizeH = ord(line1[2])
    gaborSize = ord(line1[4])
    laplaceSize = ord(line1[6])
    directions = ord(line1[5])
    codingMode = ord(line1[1])
    matchingMode = ord(line1[3])
    line2 = fin.readline().strip()
    print('line2:', len(line2))
    # TODO need to infer
    for i in range(8):
        print('line2:', ord(line2[i]))
    coding = []
    for i in range(500):
        c = []
        for j in range(500):
            c.append(ord(line2[i*500+j+8]))
        coding.append(c)
    coding = np.array(coding, dtype=np.uint8)
    print('coding:', coding)
    # cv2.imshow('coding', coding)
    # cv2.waitKey()
    cv2.imwrite(filename+'.png', coding, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    line3 = fin.readline().strip()
    for i in range(len(line3)):
        print('line3:', ord(line3[i]))
    fin.close()
    return line1+line2+line3


if __name__ == '__main__':
    edcc_api = EdccApi()

    # coding1 = analyze_coding('/Users/henan.wang/Downloads/origin_coding1')
    # coding2 = analyze_coding('/Users/henan.wang/Downloads/origin_coding2')
    # coding3 = analyze_coding('/Users/henan.wang/Downloads/origin_coding3')
    # matchScore = edcc_api.GetTwoPalmprintCodingMatchScore(bytes(coding1), bytes(coding2))
    # print('matchScore:', matchScore)
    # exit(-1)

    # images = get_images(IMAGE_DIR)
    # codings = get_codings(CODING_DIR)
    # images_and_codings = merge_images_and_codings(images, codings)
    # dump_images_and_codings(images_and_codings, DUMP_FILENAME, BASE_URL)
    # evaluate_match(images_and_codings, DIR)

    infos = get_infos(DATA_DIR, DIR)
    evaluate_match_new(infos, DIR)
    #evaluate_search_new(infos, DIR)
