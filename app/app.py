import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from wtforms import StringField, PasswordField, SubmitField
from wtforms.fields.choices import SelectField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.utils import secure_filename
import os
from datetime import timedelta
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from flask import request
import numpy as np
from prediction_model_2 import PricePredictionModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    properties = db.relationship('Property', backref='owner', lazy=True)

class Property(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    details = db.Column(db.String(200), nullable=False)
    image_filename = db.Column(db.String(200), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    #db.create_all()

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class AddPropertyForm(FlaskForm):
    details = StringField('Property Details', validators=[DataRequired()])
    image = StringField('Image', validators=[DataRequired()])
    area = StringField('Area', validators=[DataRequired()])
    bedrooms = StringField('Bedrooms', validators=[DataRequired()])
    bathrooms = StringField('Bathrooms', validators=[DataRequired()])
    stories = StringField('Stories', validators=[DataRequired()])
    mainroad = SelectField('Main Road', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    guestroom = SelectField('Guest Room', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    basement = SelectField('Basement', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    hotwaterheating = SelectField('Hot Water Heating', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    airconditioning = SelectField('Air Conditioning', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    parking = SelectField('Parking', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    courtyard = SelectField('Courtyard Area', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    furnishingstatus = SelectField('Furnishing Status', choices=[('furnished', 'Furnished'), ('unfurnished', 'Unfurnished'), ('semifurnished', 'Semi-furnished')], validators=[DataRequired()])
    submit = SubmitField('Add Property')

class PredictionForm(FlaskForm):
    bedrooms = StringField('Bedrooms', validators=[DataRequired()])
    bathrooms = StringField('Bathrooms', validators=[DataRequired()])
    area = StringField('Area (in square feet)', validators=[DataRequired()])
    submit = SubmitField('Predict')

class PropertyForm(FlaskForm):
    details = StringField('Property Details', validators=[DataRequired()])
    image = StringField('Image', validators=[DataRequired()])
    area = StringField('Area', validators=[DataRequired()])
    bedrooms = StringField('Bedrooms', validators=[DataRequired()])
    bathrooms = StringField('Bathrooms', validators=[DataRequired()])
    stories = StringField('Stories', validators=[DataRequired()])
    mainroad = SelectField('Main Road', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    guestroom = SelectField('Guest Room', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    basement = SelectField('Basement', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    hotwaterheating = SelectField('Hot Water Heating', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    airconditioning = SelectField('Air Conditioning', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    parking = SelectField('Parking', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    courtyard = SelectField('Courtyard Area', choices=[('yes', 'Yes'), ('no', 'No')], validators=[DataRequired()])
    furnishingstatus = SelectField('Furnishing Status', choices=[('furnished', 'Furnished'), ('unfurnished', 'Unfurnished'), ('semifurnished', 'Semi-furnished')], validators=[DataRequired()])
    submit = SubmitField('Add Property')
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route('/')
def opening_page():
    return render_template('opening_page.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, password=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.password == form.password.data:
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful!', 'success')
    return redirect(url_for('register'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    add_property_form = AddPropertyForm()
    prediction_form = PredictionForm()

    if add_property_form.validate_on_submit():
        details = add_property_form.details.data

        if 'image' not in request.files:
            flash('No image part', 'error')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded', 'success')

            property = Property(details=details, image_filename=filename, user_id=current_user.id)
            db.session.add(property)
            db.session.commit()

            return redirect(url_for('feature_listing'))
    return render_template('dashboard.html', add_property_form=add_property_form)
@app.route('/dashboard/add_property', methods=['GET', 'POST'])
@login_required
def add_property():
    form = AddPropertyForm()

    if form.validate_on_submit():
        details = form.details.data

        if 'image' not in request.files:
            flash('No image part', 'error')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded', 'success')

            property = Property(
                details=details,
                image_filename=filename,
                area=form.area.data,
                bedrooms=form.bedrooms.data,
                bathrooms=form.bathrooms.data,
                stories=form.stories.data,
                mainroad=form.mainroad.data,
                guestroom=form.guestroom.data,
                basement=form.basement.data,
                hotwaterheating=form.hotwaterheating.data,
                airconditioning=form.airconditioning.data,
                parking=form.parking.data,
                courtyard=form.courtyard.data,
                furnishingstatus=form.furnishingstatus.data,
                user_id=current_user.id
            )
            db.session.add(property)
            db.session.commit()

            return redirect(url_for('feature_listing'))

    return render_template('add_property.html', form=form)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/feature_listing')
@login_required
def feature_listing():
    # Assuming predicted houses have 'Predicted House' in the details field
    properties = Property.query.filter_by(user_id=current_user.id).filter(Property.details != 'Predicted House').all()
    return render_template('feature_listing.html', properties=properties)

# Other routes and configurations...
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the dataset
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load data
file_path = "D:/house_prediction(mywork)/app/Housing.csv"
data = pd.read_csv(file_path)

# Separate features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for preprocessing and model training
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'ridge__alpha': [0.1, 1.0, 10.0],  # Regularization parameter
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Model Evaluation
# Evaluate model on the test set
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Feature Engineering
# Example: Add new features like the total number of rooms or age of the property
# You can implement this based on your dataset and domain knowledge

# Further Model Refinement
# You can experiment with different algorithms, feature selection techniques, etc.

# Example function for predicting price
def predict_price(input_data):
    # Convert categorical variables to dummy variables
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Reorder columns to match training data
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Predict the price
    predicted_price = grid_search.predict(input_data)

    return predicted_price[0]
@login_required
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()  # Assuming PredictionForm is your WTForms form
    if request.method == 'POST' and form.validate_on_submit():
        area = int(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = request.form['mainroad']
        guestroom = request.form['guestroom']
        basement = request.form['basement']
        hotwaterheating = request.form['hotwaterheating']
        airconditioning = request.form['airconditioning']
        parking = int(request.form['parking'])
        prefarea = request.form['prefarea']
        furnishingstatus = request.form['furnishingstatus']

        input_data = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'parking': [parking],
            'prefarea': [prefarea],
            'furnishingstatus': [furnishingstatus]
        })

        predicted_price = predict_price(input_data)
        return render_template('prediction.html', form=form, predicted_price=predicted_price)
    return render_template('prediction.html', form=form, predicted_price=None)  # Pass None initially if no prediction has been made yet

if __name__ == '__main__':
    with app.app_context():
        #db.init_app(app)  # Initialize the database with the Flask app context
        db.create_all()

    app.run(port=5000,host='localhost',debug=True)