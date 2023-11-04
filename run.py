
from flask import Flask
#from app.ai import generator
from flask import request


app = Flask(__name__)

# Define a route for the API endpoint
'''@app.route("/ai/llava", methods=['POST'])
def stable_diff():
    return generator.llava.ask_image(request.json)'''

@app.route("/")
def stable_diff2():
    return "hello"

# Start the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0") 