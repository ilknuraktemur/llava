from app import ai
from app.ai import generator
from flask import request


@ai.route("/ai/llava", methods=["GET", "POST"])
def stable_diff():
    """Get stable diffusion outputs and send as a file"""
    return generator.llava.ask_image(request.json)