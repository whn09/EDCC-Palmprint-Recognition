#python3
import requests
import json
import time
import base64
import numpy as np
#import cv2


base_url = 'http://127.0.0.1:5000/'
encode_url = base_url+'encode'
compare_url = base_url+'compare'
search_url = base_url+'search'
test_filename1 = '../../database/Tongji/ROI/session1/00001.bmp'
test_filename2 = '../../database/Tongji/ROI/session1/00002.bmp'
test_filename3 = '../../database/Tongji/ROI/session1/00003.bmp'
test_filename4 = '../../database/Tongji/ROI/session1/00004.bmp'
test_filename5 = '../../database/Tongji/ROI/session1/00005.bmp'


def encode(filename):
    start = time.time()
    files = {"file": (filename, open(filename, "rb"))}
    res = requests.post(encode_url, data={}, files=files)
    end = time.time()
    print('encode time:', (end-start)*1000, 'ms')
    #print(res.ok)
    res_json = res.json()
    #print(json.dumps(res_json, indent=2))
    return res_json


def compare(filename1, filename2):
    start = time.time()
    files = {"file1": (filename1, open(filename1, "rb")), "file2": (filename2, open(filename2, "rb"))}
    res = requests.post(compare_url, data={}, files=files)
    end = time.time()
    print('compare time:', (end-start)*1000, 'ms')
    #print(res.ok)
    res_json = res.json()
    #print(json.dumps(res_json, indent=2))
    return res_json


def search(filename):
    start = time.time()
    files = {"file": (filename, open(filename, "rb"))}
    res = requests.post(search_url, data={}, files=files)
    end = time.time()
    print('search time:', (end-start)*1000, 'ms')
    #print(res.ok)
    res_json = res.json()
    #print(json.dumps(res_json, indent=2))
    return res_json

    
if __name__=='__main__':
    res_json = encode(test_filename1)
    # print(res_json)
    res_json = encode(test_filename2)
    # print(res_json)
    res_json = encode(test_filename3)
    # print(res_json)
    res_json = encode(test_filename4)
    # print(res_json)
    res_json = encode(test_filename5)
    #print(res_json)

    # TODO decode base64 and save image
    #img_byte = base64.b64decode(res_json['image'])
    #img_np_arr = np.fromstring(img_byte, np.uint8)
    #im = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    #cv2.imwrite('tmp.png', im)

    res_json = compare(test_filename1, test_filename2)
    print(res_json)
    res_json = compare(test_filename1, test_filename3)
    print(res_json)
    res_json = compare(test_filename1, test_filename4)
    print(res_json)
    res_json = compare(test_filename1, test_filename5)
    print(res_json)

    res_json = search(test_filename1)
    print(res_json)
    res_json = search(test_filename2)
    print(res_json)
    res_json = search(test_filename3)
    print(res_json)
    res_json = search(test_filename4)
    print(res_json)
    res_json = search(test_filename5)
    print(res_json)
