from app.main import main
from flask import request




@main.route("/")
@main.route("/index")
def index():
    return "Hello from the ubuntu machine"


@main.route("/ai/gpt")
def index1():
    return "Hello from the ubuntu machine with stable diff"