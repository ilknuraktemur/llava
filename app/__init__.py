
from flask import Flask
from app.ai import generator
from flask import request


app = Flask(__name__)

# Define a route for the API endpoint
@app.route("/ai/llava", methods=['POST'])
def stable_diff():
    """Get stable diffusion outputs and send as a file"""
    return generator.llava.ask_image(request.json)

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)