#models.py 
#contains data model for Entry 
#Entry contains fields for our AI model prediction as well as timestamp of entry
from application import db
from sqlalchemy.orm import relationship

class Entry(db.Model):
   id = db.Column(db.Integer, primary_key=True, autoincrement=True)
   userid = db.Column(db.Integer, db.ForeignKey('user_entry.userid'))
   imgPath = db.Column(db.String)
   prediction20 = db.Column(db.Integer)
   label20 = db.Column(db.String)
   accuracy20=db.Column(db.Boolean)
   prediction100 = db.Column(db.Integer)
   label100 = db.Column(db.String)
   accuracy100=db.Column(db.Boolean)
   predicted_on = db.Column(db.DateTime, nullable=False)
   user_entry = relationship("UserEntry", back_populates="entries")

class UserEntry(db.Model):
   userid = db.Column(db.Integer, primary_key=True, autoincrement=True)
   usertype = db.Column(db.String)
   username = db.Column(db.Integer)
   password = db.Column(db.String)
   entries = relationship("Entry", back_populates="user_entry")

