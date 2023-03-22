from flask_wtf import FlaskForm
from wtforms import SelectField,StringField,PasswordField,validators,SubmitField,IntegerField

class ThemeForm(FlaskForm):
   theme = SelectField(
      'Theme',
      choices=[('', 'Theme'),('light', 'Light'), ('dark', 'Dark')],
      default='',
      render_kw={'onchange': 'this.form.submit();'}
   )

   
class RegisterForm(FlaskForm):
   username = StringField(
      'Username',
      validators=[
         validators.InputRequired()
      ]
   )
   password=PasswordField(
      'Password',
      validators=[
         validators.InputRequired()
      ]
   )
   submit=SubmitField('Register')

class LoginForm(FlaskForm):
   username = StringField(
      'Username',
      validators=[
         validators.InputRequired()
      ]
   )
   password=PasswordField(
      'Password',
      validators=[
         validators.InputRequired()
      ]
   )
   submit=SubmitField('Login')

