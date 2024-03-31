from app.app import app, db  # or just 'from app import app, db' if 'app.py' is in the root folder

if __name__ == '__main__':
    with app.app_context():
        #db.init_app(app)  # Initialize the database with the Flask app context
        db.create_all()

    app.run(port=5000,host='localhost',debug=True)