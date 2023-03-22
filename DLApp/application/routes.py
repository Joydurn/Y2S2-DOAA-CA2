from application import app, db
from flask import render_template, request, flash, session,send_from_directory,url_for,redirect,jsonify
from flask_cors import cross_origin
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import re
import base64
import json
import numpy as np
import requests
import os
from application.models import Entry, UserEntry
from flask_uploads import UploadSet,configure_uploads

from datetime import datetime, timedelta
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileRequired,FileAllowed
from application.forms import ThemeForm,RegisterForm,LoginForm
from wtforms import widgets,DateField,StringField,SubmitField,DateField,MultipleFileField,ValidationError,SelectMultipleField
from wtforms.validators import Optional
from sqlalchemy import or_,func,case

#Server URLs for predictions
url1 = 'https://ca2-model-servers1.onrender.com/v1/models/cifar/versions/1:predict'
url2 = 'https://ca2-model-servers1.onrender.com/v1/models/cifar/versions/2:predict'

#DATABASE FUNCTIONS
#add an entry to db 
def add_entry(new_entry):
 try:
   db.session.add(new_entry)
   db.session.commit()
   return new_entry.id
 except Exception as error:
   db.session.rollback()
   flash(error,"danger")

#remove entry by id
def remove_entry(id):
   try:
      entry = db.get_or_404(Entry, id)
      db.session.delete(entry)
      db.session.commit()
   except Exception as error:
      db.session.rollback()
      flash(error,"danger")
      return 0

#edit accuracy by id
def edit_entry20(id,accuracy):
   try:
      print(accuracy)
      entry = db.get_or_404(Entry, id)
      entry.accuracy20 = accuracy
      db.session.commit()
      print('\n\nedited\n\n')
   except Exception as error:
      db.session.rollback()
      flash(error,"danger")
      return 0

#edit accuracy by id
def edit_entry100(id,accuracy):
   try:
      print(accuracy)
      entry = db.get_or_404(Entry, id)
      entry.accuracy100 = accuracy
      db.session.commit()
      print('\n\nedited\n\n')
   except Exception as error:
      db.session.rollback()
      flash(error,"danger")
      return 0

#import meta data for classes
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
data_meta = unpickle('application/data/metaData')
fine_labels=data_meta['fine_label_names']
coarse_labels=data_meta['coarse_label_names']
data_meta=None
def translateLabel(label,type=100):
   if type==100:
      return fine_labels[label]
   elif type==20:
      return coarse_labels[label]


photos=UploadSet('photos',extensions=('jpeg','jpg','png'))
configure_uploads(app,photos)

class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()

class UploadForm(FlaskForm):
   model=MultiCheckboxField('Model',
   choices=[
      ('20','CIFAR20 Model'),
      ('100','CIFAR100 Model')
   ], default=['20','100'],
   validators=[

   ])

   photo=FileField(
      validators=[
         FileAllowed(photos,'Only JPG/JPEG/PNG images are allowed'),
         FileRequired('Field shield should not be empty')
      ]
   )
   submit=SubmitField('Predict')




#custom validator
def multiple_file_allowed(form, field):
   for file in field.data:
      _, file_extension = os.path.splitext(file.filename)
      if not file_extension:
         raise ValidationError('Please upload one or more files first')
      if not photos.extension_allowed(file_extension[1:].lower()):
         print('yessss')
         raise ValidationError('Only JPG/JPEG/PNG images are allowed')

class UploadMultipleForm(FlaskForm):
   images=MultipleFileField(
      validators=[
         multiple_file_allowed
      ]
   )
   submit=SubmitField('Predict Batch')



def parseImage(imgData):
   # parse canvas bytes and save as output.png
   imgstr = re.search(b'base64,(.*)', imgData).group(1)
   with open('output.png','wb') as output:
      output.write(base64.decodebytes(imgstr))
   im = Image.open('output.png').convert('RGB')
   im_invert = ImageOps.invert(im)
   im_invert.save('output.png')




def make_prediction100(instances):
   data = json.dumps({"signature_name": "serving_default", "instances":
   instances.tolist()})
   headers = {"content-type": "application/json"}
   
   json_response = requests.post(url1, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

def make_prediction20(instances):
   data = json.dumps({"signature_name": "serving_default", "instances":
   instances.tolist()})
   headers = {"content-type": "application/json"}
   
   json_response = requests.post(url2, data=data, headers=headers)
   predictions = json.loads(json_response.text)['predictions']
   return predictions

#CHECK FOR LOGIN
def authorised():
   username=session.get('username',None)
   userid=session.get('userid',None)
   usertype=session.get('usertype',None)
   if username and userid and usertype:
      return True
   else:
      return False

@app.route('/', methods=['GET'])
def redirect1():
   return redirect(url_for('uploadFile'))
@app.route('/index', methods=['GET'])
def redirect2():
   return redirect(url_for('uploadFile'))
@app.route('/home', methods=['GET','POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
#HOME PAGE / SINGLE PREDICTIONS
def uploadFile():
   def predict20():
      #predictions for CIFAR20
         predictions20 = make_prediction20(img)[0]
         response20=None
         bestLabels20=np.argsort(predictions20)
         response20={}
         for i in range(1,4): #get top 3 
            label=translateLabel(bestLabels20[-i],type=20)
            print(label)
            response20[label]=predictions20[bestLabels20[-i]] #get probability
         return bestLabels20,response20 

   def predict100():
      #predictions for CIFAR100
      #predictions for cifar100
         predictions100 = make_prediction100(img)[0]
         response100=None
         bestLabels100=np.argsort(predictions100)
         response100={}
         for i in range(1,4): #get top 3 
            label=translateLabel(bestLabels100[-i],type=100)
            response100[label]=predictions100[bestLabels100[-i]] #get probability
         return bestLabels100,response100

   if not authorised():
      flash('Please login to access our services.')
      return redirect(url_for('login'))

   form=UploadForm()
   themeForm=ThemeForm()
   models=form.model.data 
   print(models)
   print(form.validate_on_submit())
   print(form.errors)
   if form.validate_on_submit():
      models=form.model.data 
      print(models)

      filename=photos.save(form.photo.data)
      file_url=url_for('get_file',filename=filename)
      file_url_full='application/static'+file_url
      img = image.img_to_array(image.load_img(file_url_full, color_mode="rgb",
         target_size=(32, 32))) 
      img = img.reshape(1,32,32,3)
      if not models:
         flash('No models selected!')
         return render_template('index.html',title='Home',themeForm=themeForm,form=form,file_url=None,prediction100=None,prediction20=None,id=None)
      elif len(models)==2: #use both models
         print('both')

         bestLabels20,response20=predict20()
         bestLabels100,response100=predict100()
         #do saving to db here 
         new_entry = Entry(
               userid=session['userid'],
               imgPath=file_url,
               prediction20=int(bestLabels20[-1]),
               label20=translateLabel(bestLabels20[-1],type=20),
               accuracy20=None,
               prediction100=int(bestLabels100[-1]),
               label100=translateLabel(bestLabels100[-1]),
               accuracy100=None,
               predicted_on=datetime.now())
         id=add_entry(new_entry)
         
      elif models[0]=='20': #cifar 20 only
         bestLabels20,response20=predict20()
         response100=None
         #do saving to db here 
         new_entry = Entry(
               userid=session['userid'],
               imgPath=file_url,
               prediction20=int(bestLabels20[-1]),
               label20=translateLabel(bestLabels20[-1],type=20),
               accuracy20=None,
               prediction100=None,
               label100=None,
               accuracy100=None,
               predicted_on=datetime.now())
         id=add_entry(new_entry)
      elif models[0]=='100': #cifar 100 only
         bestLabels100,response100=predict100()
         response20=None
         #do saving to db here 
         new_entry = Entry(
               userid=session['userid'],
               imgPath=file_url,
               prediction20=None,
               label20=None,
               accuracy20=None,
               prediction100=int(bestLabels100[-1]),
               label100=translateLabel(bestLabels100[-1]),
               accuracy100=None,
               predicted_on=datetime.now())
         id=add_entry(new_entry)
      
      else: 
         print('elseeee')
      flash('Success! See results below...')
      return render_template('index.html',title='Home Prediction Results',themeForm=themeForm,form=form,file_url=file_url,prediction100=response100,prediction20=response20,id=id)
   else:
      print('big else')
      file_url=None
      id=None
      response100=None
      response20=None
      return render_template('index.html',title='Home',themeForm=themeForm,form=form,file_url=file_url,prediction100=response100,prediction20=response20,id=None)


#BATCH PREDICTIONS PAGE
@app.route('/batch', methods=['GET','POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def uploadMultipleFiles():
   if not authorised():
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   form=UploadMultipleForm()
   themeForm=ThemeForm()
   if form.validate_on_submit():
      results20=[]
      results100=[]
      idList=[]
      filesList=[]
      for file in form.images.data:
         filename=photos.save(file)
         file_url=url_for('get_file',filename=filename)
         file_url_full='application/static'+file_url
         img = image.img_to_array(image.load_img(file_url_full, color_mode="rgb",
            target_size=(32, 32))) 
         img = img.reshape(1,32,32,3)

         #predictions for cifar100
         predictions100 = make_prediction100(img)[0]
         ret = ""
         response100=None
         bestLabels100=np.argsort(predictions100)
         response100={}
         for i in range(1,4): #get top 3 
            label=translateLabel(bestLabels100[-i],type=100)
            response100[label]=predictions100[bestLabels100[-i]] #get probability

         #predictions for CIFAR20
         predictions20 = make_prediction20(img)[0]
         ret = ""
         response20=None
         bestLabels20=np.argsort(predictions20)
         response20={}
         for i in range(1,4): #get top 3 
            label=translateLabel(bestLabels20[-i],type=20)
            # print(label)
            response20[label]=predictions20[bestLabels20[-i]] #get probability
         #do saving to db here 
         new_entry = Entry(
               userid=session['userid'],
               imgPath=file_url,
               prediction20=int(bestLabels20[-1]),
               label20=translateLabel(bestLabels20[-1],type=20),
               accuracy20=None,
               prediction100=int(bestLabels100[-1]),
               label100=translateLabel(bestLabels100[-1]),
               accuracy100=None,
               predicted_on=datetime.now())
         id=add_entry(new_entry)
         idList.append(id)
         results20.append(response20)
         results100.append(response100)
         filesList.append(file_url)


      #huge iterable for template to use
      results=enumerate(zip(filesList,results20,results100,idList))
      flash('Success! See results below...')
      return render_template('batch.html',title='Batch Results',themeForm=themeForm,form=form,results=results)
   else:
      results=None
      return render_template('batch.html',title='Batch Prediction',themeForm=themeForm,form=form,results=results)

   
#RETRIEVE IMAGES
@app.route('/upload/<filename>')
def get_file(filename):
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'],filename)

#PREDICTION HISTORY PAGE
@app.route('/predictions', methods=['GET','POST'])
def predictionsHistory():
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   #check if admin
   if not session['usertype']=='admin':
      flash('Please login as an admin to access prediction history.')
      return redirect(url_for('login'))

   #GET users
   allUserEntries=UserEntry.query.all()
   #GET labels
   Pred100Entries=list(db.session.query(Entry.prediction100, Entry.label100).distinct().all())
   Pred20Entries=list(db.session.query(Entry.prediction20, Entry.label20).distinct().all())

   print(Pred100Entries)

   #CREATE FORM WITH CHECKBOXES
   users=[('any','Any')]
   for user in allUserEntries:
      users.append(
         (str(user.__dict__['userid']),
         f"{user.__dict__['username']} ({user.__dict__['userid']})"
         )
      )
   
   labels20=[('any','Any')]
   for pred,label in Pred20Entries:
      labels20.append(
         (str(pred),
         f"({pred}) {label}"
         )
      )

   labels100=[('any','Any')]
   for pred,label in Pred100Entries:
      labels100.append(
         (str(pred),
         f"({pred}) {label}"
         )
      )

   def validate_enddate(form, field):
      if field.data and form.startdate.data:
         if field.data < form.startdate.data:
            raise ValidationError("End date must not be earlier than start date")

   class FilterForm(FlaskForm):
      search=StringField('Search Keyword')
      userid=SelectMultipleField('User (ID)',
      choices=users,
      default=['any'])

      usertype=SelectMultipleField('User Type',
         choices=[('any', 'Any'),('admin', 'Admin'),('user', 'User')],
         default=['any'],
      )

      pred20=SelectMultipleField('CIFAR20 Label',
         choices=labels20,
         default=['any'],
      )

      pred100=SelectMultipleField('CIFAR100 Label',
         choices=labels100,
         default=['any'],
      )

      startdate = DateField('Start Date', format='%Y-%m-%d', validators=[Optional()])
      enddate = DateField('End Date', format='%Y-%m-%d', validators=[Optional(),validate_enddate])

   

      submit=SubmitField('Filter')

   
   form= FilterForm()
   themeForm = ThemeForm()
   query = db.select(Entry).join(UserEntry).order_by(Entry.id)
   # results=results.query.filter(results.usertype == "admin")

   #functions for calculating accuracy
   def modifyQuery(query):
         accuracy20 = (func.sum(case(
            (Entry.accuracy20 == True, 1), 
            else_=0
         )) / func.count(Entry.id)) * 100
         accuracy100 = (func.sum(case(
            (Entry.accuracy100 == True, 1), 
            else_=0
         )) / func.count(Entry.id)) * 100

         return query.with_entities(accuracy20.label("accuracy20"),accuracy100.label("accuracy100"))

   def getAccuracy(results):
      if results:
         accuracy20 = 0
         accuracy100 = 0
         total20 = 0
         total100= 0
         for result in results2:
            if result[0] == True:
               accuracy20 += 1
            if result[1] == True:
               accuracy100 += 1
            if result[0] is not None:
               total20 += 1
            if result[1] is not None:
               total100 += 1

         if total20==0:
            accuracy20perc=None 
         else:
            accuracy20perc=accuracy20 / total20 * 100
         if total100==0:
            accuracy100perc=None
         else:
            accuracy100perc=accuracy100 / total100 * 100
         return accuracy20perc,accuracy100perc
      else:
         return None, None


   if not form.validate_on_submit():
      results = db.session.execute(query).scalars().all()
      query = Entry.query.from_statement(query)
      results2 = db.session.execute(modifyQuery(query)).all()
      accuracy20_percent, accuracy100_percent = getAccuracy(results2)
      print(accuracy20_percent)
      print(accuracy100_percent)

      print(results2)
      return render_template("predictions.html",title='Predictions History',themeForm=themeForm,
         form=form,entries=results,accuracy100=accuracy100_percent,accuracy20=accuracy20_percent)
   else:
      search=form.search.data
      userid=form.userid.data
      usertype=form.usertype.data

      pred20=form.pred20.data 
      pred100=form.pred100.data

      startdate=form.startdate.data 
      enddate=form.enddate.data 
      print('\n'*5)
      print(startdate,enddate)
      print('\n'*5)
      if search:
         print('searching')
         query=query.filter(or_(
               Entry.id.like(f'%{search}%'),
               Entry.userid.like(f'%{search}%'),
               UserEntry.username.like(f'%{search}%'),
               UserEntry.usertype.like(f'%{search}%'),
               Entry.imgPath.like(f'%{search}%'),
               Entry.label20.like(f'%{search}%'),
               Entry.label100.like(f'%{search}%'),
               Entry.prediction20.like(f'%{search}%'),
               Entry.prediction100.like(f'%{search}%'),
               Entry.predicted_on.like(f'%{search}%'),
         ))
      if userid and 'any' not in userid:
         print('filtering stuff')
         userid=[int(x) for x in userid]
         query=query.where(Entry.userid.in_(userid))
      if usertype and 'any' not in usertype:
         print('filtering stuff2')
         query=query.where(UserEntry.usertype.in_(usertype))
      
      if pred20 and 'any' not in pred20:
         print('filtering3')
         pred20=[int(x) for x in pred20]
         query=query.where(Entry.prediction20.in_(pred20))
      if pred100 and 'any' not in pred100:
         print('filtering4')
         pred100=[int(x) for x in pred100]
         query=query.where(Entry.prediction100.in_(pred100))
      
      if startdate:
         query = query.where(Entry.predicted_on >= startdate - timedelta(days=1))

      if enddate:
         query = query.where(Entry.predicted_on <= enddate + timedelta(days=1))
      print('\nerrors:',form.errors)
      #final query
      results = db.session.execute(query).scalars().all()

      query = Entry.query.from_statement(query)
      #get accuracy value
      
      results2 = db.session.execute(modifyQuery(query)).all()
      
      accuracy20_percent, accuracy100_percent = getAccuracy(results2)
      print(accuracy20_percent)
      print(accuracy100_percent)
      return render_template("predictions.html",title='Predictions History',themeForm=themeForm,
         form=form,entries=results,accuracy100=accuracy100_percent,accuracy20=accuracy20_percent)
      


#API for removing entries, used in predictions history
@app.route('/remove', methods=['POST'])
def remove():
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   #check if admin
   if not session['usertype']=='admin':
      flash('Please login as an admin to access prediction history.')
      return redirect(url_for('login'))
   req = request.form
   id = req["id"]
   remove_entry(id)
   return redirect(request.referrer)

#API for editing entries' accuracy20
@app.route('/edit20/<id>', methods=['POST'])
def edit20(id):
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   req = request.form
   accuracy = req["accuracy"]
   if accuracy == 'True':
      accuracy = True
   elif accuracy == 'False':
      accuracy = False

   edit_entry20(id,accuracy)
   print(request.referrer)
   last_word = request.referrer.split("/")[-1]
   print(last_word)
   if last_word=='home' or last_word=='batch':
      return render_template('success.html',themeForm=ThemeForm())
   return redirect(request.referrer)

#API for editing entries' accuracy100
@app.route('/edit100/<id>', methods=['POST'])
def edit100(id):
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   req = request.form
   accuracy = req["accuracy"]
   if accuracy == 'True':
      accuracy = True
   elif accuracy == 'False':
      accuracy = False
   edit_entry100(id,accuracy)

   if request.referrer==url_for('uploadFile') or request.referrer== url_for('uploadMultipleFiles'):
      return render_template('success.html',themeForm=ThemeForm())
   return redirect(request.referrer)
   

#theme switch
@app.route('/update_theme', methods=['POST'])
def update_theme():
    themeForm = ThemeForm()
    if themeForm.validate_on_submit():
        session['theme'] = themeForm.theme.data
    return redirect(request.referrer)

#REGISTRATION
@app.route('/register',methods=['GET','POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def register():
   themeForm=ThemeForm()
   form=RegisterForm()
   if form.validate_on_submit():
      username=form.username.data 
      password=form.password.data
      # query=f'SELECT * from user_entry WHERE username="{username}"'
      # result = db.engine.execute(query)
      user = db.session.query(UserEntry).filter_by(username=username).first()
      if user:
         flash('User already exists!','danger')
         print('flashing')
      else:
         # Create new user entry
         new_user = UserEntry(username=username, password=password,usertype='user')
         db.session.add(new_user)
         db.session.commit()
         flash('User created successfully! Please log in to continue')
         return redirect(url_for('login'))

   return render_template('register.html',themeForm=themeForm,form=form)

#LOGIN
@app.route('/login',methods=['GET','POST'])
def login():
   themeForm=ThemeForm()
   form=LoginForm()
   print('logining')
   if form.validate_on_submit():
      username=form.username.data 
      password=form.password.data
      user = db.session.query(UserEntry).filter_by(username=username,password=password).first()
      if user:
         session['username']=username 
         session['userid']=user.userid
         session['usertype']=user.usertype
         flash('Logged in successfully!')
         print('logged in!')
         return redirect(f'/user/{username}')
      else:
         flash('Invalid username or password, please try again','danger')
   return render_template('login.html',themeForm=themeForm,form=form)

#USER DASHBOARD AND HISTORY
@app.route('/user/<username>',methods=['GET','POST'])
def user_dashboard(username): # e must be in there
   if not authorised(): #not logged in
      flash('Please login to access our services.')
      return redirect(url_for('login'))
   #check username
   if not session['username']==username:
      flash('Invalid username/Not authorised')
      return redirect(url_for('login'))
   
   title=f"{username}'s Dashboard"


   #GET labels
   Pred100Entries=list(db.session.query(Entry.prediction100, Entry.label100).distinct().all())
   Pred20Entries=list(db.session.query(Entry.prediction20, Entry.label20).distinct().all())

   print(Pred100Entries)

   
   
   labels20=[('any','Any')]
   for pred,label in Pred20Entries:
      labels20.append(
         (str(pred),
         f"({pred}) {label}"
         )
      )

   labels100=[('any','Any')]
   for pred,label in Pred100Entries:
      labels100.append(
         (str(pred),
         f"({pred}) {label}"
         )
      )

   def validate_enddate(form, field):
      if field.data and form.startdate.data:
         if field.data < form.startdate.data:
            raise ValidationError("End date must not be earlier than start date")

   class FilterForm(FlaskForm):
      search=StringField('Search Keyword')
      

      pred20=SelectMultipleField('CIFAR20 Label',
         choices=labels20,
         default=['any'],
      )

      pred100=SelectMultipleField('CIFAR100 Label',
         choices=labels100,
         default=['any'],
      )

      startdate = DateField('Start Date', format='%Y-%m-%d', validators=[Optional()])
      enddate = DateField('End Date', format='%Y-%m-%d', validators=[Optional(),validate_enddate])

   

      submit=SubmitField('Filter')

   
   form= FilterForm()
   themeForm = ThemeForm()

   query = db.select(Entry).join(UserEntry).filter(Entry.userid == session['userid']).order_by(Entry.id)

   #functions for calculating accuracy
   def modifyQuery(query):
         accuracy20 = (func.sum(case(
            (Entry.accuracy20 == True, 1), 
            else_=0
         )) / func.count(Entry.id)) * 100
         accuracy100 = (func.sum(case(
            (Entry.accuracy100 == True, 1), 
            else_=0
         )) / func.count(Entry.id)) * 100

         return query.with_entities(accuracy20.label("accuracy20"),accuracy100.label("accuracy100"))

   def getAccuracy(results):
      if results:
         accuracy20 = 0
         accuracy100 = 0
         total20 = 0
         total100= 0
         for result in results2:
            if result[0] == True:
               accuracy20 += 1
            if result[1] == True:
               accuracy100 += 1
            if result[0] is not None:
               total20 += 1
            if result[1] is not None:
               total100 += 1

         if total20==0:
            accuracy20perc=None 
         else:
            accuracy20perc=accuracy20 / total20 * 100
         if total100==0:
            accuracy100perc=None
         else:
            accuracy100perc=accuracy100 / total100 * 100
         return accuracy20perc,accuracy100perc
      else:
         return None, None


   if not form.validate_on_submit():
      results = db.session.execute(query).scalars().all()
      query = Entry.query.from_statement(query)
      results2 = db.session.execute(modifyQuery(query)).all()
      accuracy20_percent, accuracy100_percent = getAccuracy(results2)
      results2=None
      return render_template("user.html",title=title,themeForm=themeForm,
         form=form,entries=results,accuracy100=accuracy100_percent,accuracy20=accuracy20_percent)
   else:
      search=form.search.data

      pred20=form.pred20.data 
      pred100=form.pred100.data

      startdate=form.startdate.data 
      enddate=form.enddate.data 
      print('\n'*5)
      print(startdate,enddate)
      print('\n'*5)
      if search:
         print('searching')
         query=query.filter(or_(
               Entry.id.like(f'%{search}%'),
               Entry.userid.like(f'%{search}%'),
               UserEntry.username.like(f'%{search}%'),
               UserEntry.usertype.like(f'%{search}%'),
               Entry.imgPath.like(f'%{search}%'),
               Entry.label20.like(f'%{search}%'),
               Entry.label100.like(f'%{search}%'),
               Entry.prediction20.like(f'%{search}%'),
               Entry.prediction100.like(f'%{search}%'),
               Entry.predicted_on.like(f'%{search}%'),
         ))
      
      if pred20 and 'any' not in pred20:
         print('filtering3')
         pred20=[int(x) for x in pred20]
         query=query.where(Entry.prediction20.in_(pred20))
      if pred100 and 'any' not in pred100:
         print('filtering4')
         pred100=[int(x) for x in pred100]
         query=query.where(Entry.prediction100.in_(pred100))
      
      if startdate:
         query = query.where(Entry.predicted_on >= startdate - timedelta(days=1))

      if enddate:
         query = query.where(Entry.predicted_on <= enddate + timedelta(days=1))
      print('\nerrors:',form.errors)
      #final query
      results = db.session.execute(query).scalars().all()

      query = Entry.query.from_statement(query)
      #get accuracy value
      
      results2 = db.session.execute(modifyQuery(query)).all()
      
      accuracy20_percent, accuracy100_percent = getAccuracy(results2)
      print(accuracy20_percent)
      print(accuracy100_percent)
      return render_template("user.html",title=title,themeForm=themeForm,
         form=form,entries=results,accuracy100=accuracy100_percent,accuracy20=accuracy20_percent)
   

   
   # themeForm = ThemeForm()
   # return render_template('user.html',themeForm=themeForm)

#LOGOUT
@app.route('/logout',methods=['GET','POST'])
def logout():
   if not authorised(): #not logged in
      flash('You must be logged in to log out ^_^')
      return redirect(url_for('login'))
   session['username']=None
   session['userid']=None
   session['usertype']=None
   flash('Logged out successfully!')
   print('logged out!')
   return redirect('/login')

#page not found
@app.errorhandler(404)
def page_not_found(e): # e must be in there
   themeForm = ThemeForm()
   # note that we set the 404 status, this is what it catches
   return render_template('404.html',themeForm=themeForm), 404

#wrong method
@app.errorhandler(405)
def page_not_found(e): # e must be in there
   themeForm = ThemeForm()
   return render_template('405.html',themeForm=themeForm), 405


# ****************************** API TESTS ********************

#get one entry by id
def get_entry(id):
   try:
      result = db.get_or_404(Entry, id)
      return result
   except Exception as error:
      db.session.rollback()
      flash(error,"danger")
      return 0



#API: add entry
@app.route("/api/add", methods=['POST'])
def api_add():
   #retrieve the json file posted from client
   data = request.get_json()
   #retrieve each field from the data
   userid = int(data['userid'])
   imgPath = str(data['imgPath'])
   prediction20 = int(data['prediction20'])
   label20 = str(data['label20'])
   accuracy20 = data['accuracy20']
   prediction100 = int(data['prediction100'])
   label100 = str(data['label100'])
   accuracy100 = data['accuracy100']
   #create an Entry object store all data for db action
   new_entry = Entry(
            userid=userid,
            imgPath=imgPath,
            prediction20 =prediction20,
            label20=label20,
            accuracy20=accuracy20,
            prediction100=prediction100,
            label100=label100,
            accuracy100=accuracy100,
            predicted_on=datetime.now())
   #invoke the add entry function to add entry
   result = add_entry(new_entry)
   #return the result of the db action
   return jsonify({'id':result})

#API delete entry
@app.route("/api/delete/<id>", methods=['GET'])
def api_delete(id):
   entry = remove_entry(int(id))
   return jsonify({'result':'ok'})


#API get entry
@app.route("/api/get/<id>", methods=['GET'])
def api_get(id):
   #retrieve the entry using id from client
   entry = get_entry(int(id))
   #Prepare a dictionary for json conversion
   data = {'id' : entry.id,
   'userid' : entry.userid,
   'imgPath' : entry.imgPath,
   'prediction20' : entry.prediction20,
   'label20' : entry.label20,
   'accuracy20' : entry.accuracy20,
   'prediction100' : entry.prediction100,
   'label100' : entry.label100,
   'accuracy100' : entry.accuracy100,
   'predicted_on': entry.predicted_on}
   #Convert the data to json
   result = jsonify(data)
   return result #response back

