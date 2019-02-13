#!/usr/bin/env python
#Ю -*- coding: utf-8 -*-

from functools import wraps
from flask import Flask, g, make_response, request, current_app
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
from models import db, User
import click
import datetime
import calendar
from dateutil.relativedelta import relativedelta
import jwt
import json


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}}, support_credentials=True, methods=['GET', 'POST', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'])
api = Api(app, prefix="/api/v1")
auth = HTTPBasicAuth()
app.config.from_envvar('APPSETTINGS')
app.config['PROPAGATE_EXCEPTIONS'] = True
db.init_app(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)


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
        data = jwt.decode(token, current_app.config['SECRET_KEY'])
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
    print("New user added", newuser)

                
if __name__ == '__main__':
    app.run(host='0.0.0.0')
