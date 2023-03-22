from flask import Flask 
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy 

import os

#Create database
db = SQLAlchemy()

#create the Flask app
app = Flask(__name__)
CORS(app)
# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('application','static', 'upload')

app.config.from_pyfile('config.cfg')
app.config['UPLOADED_PHOTOS_DEST'] = UPLOAD_FOLDER



with app.app_context():
   db.init_app(app)
   from .models import Entry,UserEntry
   db.create_all()
   db.session.commit()
   print('Created Database!')



#Flask form for upload image


# # Define allowed files 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from application import routes
