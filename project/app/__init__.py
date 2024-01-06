from flask import Flask

app = Flask('my_cnn_model')

# Import the routes
from app import routes
