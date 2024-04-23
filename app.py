import io
import json                    
import base64                  
import logging    
import jwt         
import numpy as np
from PIL import Image
import torchvision.transforms as trfm
import torch
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from utils import load_weight, _is_same, detect_face
from datetime import datetime
import models.facenet
from functools import wraps
from auth import AUTH_DATA
from  flask_login import LoginManager, login_user, current_user, logout_user

from flask import Flask, request, jsonify, abort, session, make_response, render_template, redirect, url_for

app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)
app.config['SECRET_KEY'] = 'thisisthesecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://wpgguaeodrkzxp:24145f3b4d0fb381d21188305a992f38792d096dbb07b43374e7601cf623591d@ec2-44-209-24-62.compute-1.amazonaws.com:5432/deeonte39ngmo8'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)          
app.logger.setLevel(logging.DEBUG)
app.config['SECRET_KEY'] = 'thisisthesecretkey'

login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class User(UserMixin, db.Model):
  id = db.Column(db.Integer, primary_key=True)
  username = db.Column(db.String(50), index=True, unique=True)
  email = db.Column(db.String(150), unique = True, index = True)
  country = db.Column(db.String(50))
  password_hash = db.Column(db.String(150))
  joined_at = db.Column(db.DateTime(), default = datetime.utcnow, index = True)

  def __repr__(self):
    return f"Name - {self.username}, Email - {self.email}" 

  def set_password(self, password):
        self.password_hash = generate_password_hash(password)

  def check_password(self,password):
      return check_password_hash(self.password_hash,password)

@app.route("/",methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username and password:
            if username in AUTH_DATA:
                if AUTH_DATA[username] == password:
                    session['username']= username
                    return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('login'))

@app.route('/home')
def index():
    login=False
    if 'username' in session:
        login=True
        token = jwt.encode({'user':session['username']},app.config['SECRET_KEY'])
        token = token.decode('UTF-8')
    return render_template('login_home.html',login=login,token=token)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.json['token']
        if not token:
            return jsonify({'message':'Token is Missing'}), 403
        try:
            data = jwt.decode(token,app.config['SECRET_KEY'])
        except:
            return jsonify({'message':'Token is wrong or invalid'})
        user_present = User.query.filter_by(username = data['user']).first()
        if user_present == None:
            return jsonify({'message':'You are not registered as our user'}), 403
        global user
        user = data['user']
        return f(*args,**kwargs)
    return decorated




@app.route("/facerec", methods=['POST'])
@token_required
def facerec():         
    #global user     
    if not request.json or 'image1' not in request.json or 'image2' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im1_b64 = request.json['image1']
    im2_b64 = request.json['image2']

    # convert it into bytes  
    img1_bytes = base64.b64decode(im1_b64.encode('utf-8'))
    img2_bytes = base64.b64decode(im2_b64.encode('utf-8'))


    # convert bytes data to PIL Image object
    img1 = Image.open(io.BytesIO(img1_bytes))
    img2 = Image.open(io.BytesIO(img2_bytes))

    cache_key = "NN1_BN_FaceNet_2K_160"
    default_threshold = 0.70
    trfrm = trfm.Compose([trfm.Resize((160, 160)),
                                             trfm.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getattr(models.facenet, cache_key)(embedding_size=128)
    model.to(device)
    model = load_weight(model, cache_key, device)
    face1 = detect_face(img1)
    face2 = detect_face(img2)
    distance, output = _is_same(face1,face2,model,trfrm,default_threshold) 

    

    result_dict = {'user':user, 'distance':distance,'threshold':default_threshold, 'output': output}
    return result_dict
  

  
  
if __name__ == "__main__":     
    app.run(debug=True, host='0.0.0.0')

