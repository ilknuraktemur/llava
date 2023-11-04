cd ..
sudo apt update
python3 -m venv llava
source llava/bin/activate
sudo apt -y install nginx
pip install -r requirements.txt
sudo mv flask_app /etc/nginx/sites-enabled/
sudo unlink /etc/nginx/sites-enabled/default
sudo nginx -t
sudo nginx -s reload
sudo ufw allow 4000
gunicorn -b 0.0.0.0:4000 --workers=1 --timeout 120 run:app