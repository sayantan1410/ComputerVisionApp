import logging
import sys
logging.basicConfig(stream=sys.stderr)
activate_this = "/home/sayantan/venv/bin/activate_this.py"
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__ = activate_this))
sys.path.insert(0, '/var/www/prod_fnet_api')
from app import app as application
application.secret_key = 'thisisthesecretkey'