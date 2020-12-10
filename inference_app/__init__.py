import os
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__, instance_relative_config=False)
    CORS(app)
    with app.app_context():
        from . import routes
        return app