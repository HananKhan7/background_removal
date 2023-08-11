# Importing libraries
from flask import Flask
from router import router

app = Flask(__name__)
app.register_blueprint(router)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')