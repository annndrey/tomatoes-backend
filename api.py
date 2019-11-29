#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import wraps
from flask import Flask, g, make_response, request, current_app
from flask import abort as fabort
from flask_restful import Resource, Api, reqparse, abort, marshal_with
from flask.json import jsonify
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import or_
from sqlalchemy import func as sql_func
from flask_marshmallow import Marshmallow
from flask_httpauth import HTTPBasicAuth
from flask_cors import CORS, cross_origin
from flask_restful.utils import cors
from marshmallow import fields
from marshmallow_enum import EnumField
from models import db, User, UserQuery

import os
import uuid
import logging

import click
import datetime
import calendar
from dateutil.relativedelta import relativedelta
import jwt
import json


import io
import requests
from PIL import Image
import glob
from collections import OrderedDict
import cv2
import numpy as np  

# from predict.py
# from maskrcnn_benchmark.config import cfg
# import predictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

from predict2 import create_predict_instance


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}, support_credentials=True, methods=['GET', 'POST', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'])
api = Api(app, prefix="/api/v1")
auth = HTTPBasicAuth()
app.config.from_envvar('APPSETTINGS')
app.config['PROPAGATE_EXCEPTIONS'] = True
db.init_app(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)

MODE = app.config['MODE']
CONFIG_FILE = app.config['CONFIG_FILE']
CF_THRESHOLD = app.config['CF_THRESHOLD']
MIN_IMAGE_SIZE = app.config['MIN_IMAGE_SIZE']
BLOCKTIME = app.config['BLOCKTIME']
BLOCKREQUESTS = app.config['BLOCKREQUESTS']
FILE_PATH = app.config['FILE_PATH']
# coco_demo = None
predictor = None
leaf_metadata = None

#if __name__ != "__main__":
#    gunicorn_logger = logging.getLogger('gunicorn.error')
#    app.logger.handlers = gunicorn_logger.handlers
#    app.logger.setLevel(gunicorn_logger.level)


@app.before_first_request
def createpredictor():
    global predictor
    global leaf_metadata
    predictor, leaf_metadata = create_predict_instance()

def token_required(f):  
    @wraps(f)
    def _verify(*args, **kwargs):
        auth_headers = request.headers.get('Authorization', '').split()

        invalid_msg = {
            'message': 'Invalid token. Registeration and / or authentication required',
            'authenticated': False
        }
        expired_msg = {
            'message': 'Expired token. Reauthentication required.',
            'authenticated': False
        }

        if len(auth_headers) != 2:
            return abort(403)

        #try:
        token = auth_headers[1]
        #try:
        data = jwt.decode(token, current_app.config['SECRET_KEY'], options={'verify_exp': False})
        #except:
        #    jwt.exceptions.ExpiredSignatureError:
        #    abort(403, message="")
        user = User.query.filter_by(login=data['sub']).first()

        if not user:
            abort(404)

        return f(*args, **kwargs)

    return _verify


def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not getattr(func, 'authenticated', True):
            return func(*args, **kwargs)

        acct = basic_authentication()  # custom account lookup function

        if acct:
            return func(*args, **kwargs)

        abort(401)
    return wrapper


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)

@auth.verify_password
def verify_password(username_or_token, password):
    user = User.verify_auth_token(username_or_token)
    if not user:
        # try to authenticate with username/password
        user = User.query.filter_by(login = username_or_token).first()
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True


@app.route('/api/v1/token', methods=['POST'])
@cross_origin(supports_credentials=True)
def get_auth_token_post():
    username = request.json.get('username')
    password = request.json.get('password')
    user = User.query.filter_by(login = username).first()
    if user:
        if user.verify_password(password):
            token = user.generate_auth_token()
            response = jsonify({ 'token': "%s" % token.decode('utf-8'), "user_id":user.id, "login": user.login, "name": user.name })
            return response
    abort(404)


@app.route('/api/v1/token', methods=['GET'])
def get_auth_token():
    token = g.user.generate_auth_token()
    return jsonify({ 'token': "%s" % token })


# SCHEMAS 
class UserSchema(ma.ModelSchema):
    class Meta:
        model = User


class StatsAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
    
    def options(self, *args, **kwargs):
        return jsonify([])

    @token_required
    @cross_origin()
    def get(self, id=None):
        auth_headers = request.headers.get('Authorization', '').split()
        token = auth_headers[1]
        data = jwt.decode(token, current_app.config['SECRET_KEY'])
        user = User.query.filter_by(login=data['sub']).first()
        if not user:
            abort(404)

        return jsonify({})

    @token_required
    @cross_origin()
    def post(self):
        auth_headers = request.headers.get('Authorization', '').split()
        token = auth_headers[1]
        udata = jwt.decode(token, current_app.config['SECRET_KEY'])
        user = User.query.filter_by(login=udata['sub']).first()
        fpath = os.path.join(FILE_PATH, user.login)
        maxqueryage = current_app.config['QUERY_AGE']
        remoteip = request.remote_addr
        if not user:
            return abort(403)
        app.logger.debug(request.form)
        cf_threshold = request.form.get('inputThreshold', None)
        min_image_size = request.form.get('inputMinSize', None)
        pred_mode = request.form.get('inputMode', None)
        
        index = request.form.get('index', None)
        orig_name = request.form.get('filename', None)
        f = request.files.get('croppedfile', None)
        if not any((index, orig_name, f)):
            fabort(400, 'Index, filename & croppedfile are required')
        data = f.read()
        fsize = len(data)
        imgext = os.path.splitext(f.filename)[-1].lower()
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        fuuid = str(uuid.uuid4())
        fname = fuuid + imgext
        prevquery = db.session.query(UserQuery).filter(UserQuery.orig_name == orig_name).filter(UserQuery.fsize == fsize).filter(UserQuery.user == user).first()

        # if it was more than 3 requests in last 10 minutes,
        # return 429 too many requests
        now = datetime.datetime.now()
        sometimebefore = now - datetime.timedelta(minutes=BLOCKTIME)
        
        recentrequests = db.session.query(UserQuery).filter(UserQuery.user == user).filter(UserQuery.ipaddr == remoteip).filter(UserQuery.timestamp > sometimebefore).all()
        numrequests = len(recentrequests)
        if numrequests >= BLOCKREQUESTS:
            app.logger.info(f"It was {numrequests} requests in last {BLOCKTIME} minutes")
            app.logger.info('Too many requests')
            abort(429, message='Too many requests, try again later')

        if prevquery and prevquery.queryage <= maxqueryage:
            # return existing data without calculating
            app.logger.info("PREV REQUESTS")
            app.logger.info("Serving saved results")
            resp = json.loads(prevquery.result)
            print(prevquery.local_name)
            strippedpath = os.path.join(user.login, prevquery.local_name)
            resurl = request.host_url + strippedpath
        else:
            app.logger.info("NEW RESULTS")
            fullpath = os.path.join(fpath, fname)
            app.logger.debug(fullpath)
            with open(fullpath, 'wb') as outf:
                outf.write(data)
                
            # pil_image = Image.open(fullpath).convert("RGB")
            # np_image = np.array(pil_image)[:, :, [2, 1, 0]]
            # predict_image = coco.run_on_opencv_image(np_image)
            # predict_image=Image.fromarray(predict_image[:, :, [2, 1, 0]])
            # predict_image.save(fullpath)
            im = cv2.imread(fullpath)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=leaf_metadata, 
                           scale=1, 
                           instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(fullpath,v.get_image()[:, :, ::-1])

            result = {'leave_prediction': 'success'}
            app.logger.info(f'saving query {remoteip} {user}')
            newquery = UserQuery(local_name=fname, orig_name=orig_name, user=user, ipaddr=remoteip, result=json.dumps(result), fsize=fsize)
            db.session.add(newquery)
            db.session.commit()
            # Should return saved file url
            strippedpath = fullpath.replace(FILE_PATH, '')
            app.logger.debug(strippedpath)
            app.logger.debug(request.host_url)
            resurl = request.host_url + strippedpath[1:]
        app.logger.debug(request.host_url)
        resurl = resurl.replace('http://', 'https://')
        return jsonify({'url':resurl, 'index': index})


class UserAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.schema = UserSchema(exclude=['password_hash',])
        self.m_schema = UserSchema(many=True, exclude=['password_hash',])
        self.method_decorators = []

        
    def options(self, *args, **kwargs):
        return jsonify([])
        
    @token_required
    @cross_origin()
    def get(self, id=None):
        if not id:
            users = db.session.query(User).all()
            return jsonify(self.m_schema.dump(users).data)
        else:
            user = User.query.filter_by(id=id).first()
            if user:
                return jsonify(self.schema.dump(user).data), 200
            else:
                abort(404)

    @token_required
    @cross_origin()
    def patch(self, id):
        if not request.json:
            abort(400, message="No data provided")
            
        user = db.session.query(User).filter(User.id==id).first()
        if user:
            for attr in ['login', 'phone', 'name', 'note', 'is_confirmed', 'confirmed_on', 'password']:
                val = request.json.get(attr)
                if attr == 'password' and val:
                    user.hash_password(val)
                    
                elif attr == 'confirmed_on':
                    val = datetime.datetime.now()

                        
                if val:
                    setattr(user, attr, val)
                
            db.session.add(user)
            db.session.commit()
            return jsonify(self.schema.dump(user).data), 201
        
        abort(404, message="Not found")

    @token_required
    @cross_origin()
    def post(self):
        if not request.json:
            abort(400, message="No data provided")
        login = request.json.get('login')
        phone = request.json.get('phone')
        name = request.json.get('name')
        password = request.json.get('password')
        
        if not(any([login, phone, name])):
            return abort(400, 'Provide required fields for phone, name or login')
        
        prevuser = db.session.query(User).filter(User.login==login).first()
        if prevuser:
            abort(409, message='User exists')
        note = request.json.get('note')

        is_confirmed = request.json.get('is_confirmed')
        confirmed_on = None
        if is_confirmed:
            confirmed_on = datetime.datetime.today()

        newuser = User(login=login, is_confirmed=is_confirmed, confirmed_on=confirmed_on, phone=phone, name=name, note=note)
        
        newuser.hash_password(password)
        db.session.add(newuser)
        db.session.commit()
        
        return jsonify(self.schema.dump(newuser).data), 201
    
    def delete(self, id):
        if not id:
            abort(404, message="Not found")
        user = db.session.query(User).filter(User.id==id).first()
        if user:
            db.session.delete(user)
            db.session.commit()
            return make_response("User deleted", 204)
        abort(404, message="Not found")

api.add_resource(UserAPI, '/users', '/users/<int:id>', endpoint = 'users')
api.add_resource(StatsAPI, '/loadimage', endpoint = 'loadimage')


@app.cli.command()
@click.option('--login',  help='user@mail.com')
@click.option('--password',  help='password')
@click.option('--name',  help='name')
@click.option('--phone',  help='phone')
def adduser(login, password, name, phone):
    """ Create new user"""
        
    newuser = User(login=login, name=name, phone=phone)
    newuser.hash_password(password)
    newuser.is_confirmed = True
    newuser.confirmed_on = datetime.datetime.today()
    db.session.add(newuser)
    db.session.commit()

    app.logger.info("New user added", newuser)

                
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
