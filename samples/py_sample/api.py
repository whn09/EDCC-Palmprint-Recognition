# -*- coding: utf-8 -*-
import os
import cv2
import time
import base64
import cherrypy
from paste.translogger import TransLogger
from flask import Flask, request, abort, jsonify, make_response
from run_sample import *
from show_palmprint import get_contours


# app
app = Flask(__name__)
BAD_REQUEST = 400
STATUS_OK = 200
NOT_FOUND = 404
SERVER_ERROR = 500


@app.errorhandler(BAD_REQUEST)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), BAD_REQUEST)


@app.errorhandler(NOT_FOUND)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), NOT_FOUND)


@app.errorhandler(SERVER_ERROR)
def server_error(error):
    return make_response(jsonify({'error': 'Server Internal Error'}), SERVER_ERROR)


def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'log.access_file': 'access.log',
        'log.error_file': 'cherrypy.log',
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0',
        'server.thread_pool': 50,  # 10 is default
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()


@app.route('/')
def index():
    return 'Yeah, yeah, I highly recommend it'


@app.route('/encode', methods=['POST'])
def encode():
    #print('request.files:', len(request.files))
    if 'file' not in request.files:
        abort(BAD_REQUEST)
    file = request.files['file']
    if file.filename == '':
        abort(BAD_REQUEST)
    encoding = ''
    if file:
        filename = str(os.path.join('upload/', file.filename.split('/')[-1]))
        file.save(filename)
        outfilename = str(filename+'.out.png')
        #cherrypy.log('filename:' + filename)
        start = time.time()
        edcc_api.GetEnhanceImage(filename, outfilename, config_path)
        end = time.time()
        cherrypy.log('GetEnhanceImage time:' + str(end-start))
        encoding, codingLen = edcc_api.GetEDCCCoding(filename, config_path)
        cherrypy.log('encoding:' + encoding)
        #cherrypy.log('codingLen:' + str(codingLen))
        end2 = time.time()
        cherrypy.log('GetEDCCCoding time:' + str(end2 - end))
        palmprint_filename = get_contours(outfilename)
        #cherrypy.log('palmprint_filename:' + palmprint_filename)
        end3 = time.time()
        cherrypy.log('get_contours time:' + str(end3 - end2))

    if encoding != '':
        result = {'encoding': base64.b64encode(encoding).decode('utf-8'), 'image': base64.b64encode(cv2.imread(palmprint_filename)).decode('utf-8')}
    else:
        result = {}

    return make_response(jsonify(result), STATUS_OK)


@app.route('/compare', methods=['POST'])
def compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        abort(BAD_REQUEST)
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '' or file2.filename == '':
        abort(BAD_REQUEST)
    similarity = 0.0
    if file1 and file2:
        filename1 = str(os.path.join('upload/', file1.filename.split('/')[-1]))
        file1.save(filename1)
        filename2 = str(os.path.join('upload/', file2.filename.split('/')[-1]))
        file2.save(filename2)
        encoding1, codingLen1 = edcc_api.GetEDCCCoding(filename1, config_path)
        encoding2, codingLen2 = edcc_api.GetEDCCCoding(filename2, config_path)
        start = time.time()
        similarity = edcc_api.GetTwoPalmprintCodingMatchScore(encoding1, encoding2)
        end = time.time()
        cherrypy.log('GetTwoPalmprintCodingMatchScore time:' + str(end - start))

    is_same = False
    if similarity >= threshold:
        is_same = True
    result = {'similarity': similarity, 'is_same': is_same}

    return make_response(jsonify(result), STATUS_OK)


@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        abort(BAD_REQUEST)
    file = request.files['file']
    if file.filename == '':
        abort(BAD_REQUEST)
    max_similarity = 0
    max_palmprintcode = None
    if file:
        filename = str(os.path.join('upload/', file.filename.split('/')[-1]))
        file.save(filename)
        encoding, codingLen = edcc_api.GetEDCCCoding(filename, config_path)

        max_similarity = -1
        max_palmprintcode = None
        # TODO should have index
        cherrypy.log('sample.palmprintcodelist:' + str(len(sample.palmprintcodelist)))
        start = time.time()
        for trainPalmprintCode in sample.palmprintcodelist:
            similarity = edcc_api.GetTwoPalmprintCodingMatchScore(encoding, trainPalmprintCode.code)
            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                max_palmprintcode = trainPalmprintCode.palmprint
        end = time.time()
        cherrypy.log('search time:' + str(end - start))
    if max_palmprintcode != None:
        result = {'similarity': max_similarity, 'id': max_palmprintcode.id, 'image_path': max_palmprintcode.imagePath, 'instance_id': max_palmprintcode.instanceID}
    else:
        result = {}

    return make_response(jsonify(result), STATUS_OK)


sample = EDCCSample()
sample.readDB()
if len(sample.palmprintcodelist) == 0: # TODO bug, should only execute once
    sample.initDB()
    sample.readDB()
edcc_api = sample.get_edcc_api()
config_path = sample.get_config_path()
threshold = 0.5

if __name__ == '__main__':
    run_server()
