import logging

from dask.distributed import Client, LocalCluster
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

from src.utils import check_socket

logger = logging.getLogger(__name__)

# Initialize the database extension (SQLAlchemy) without binding it to a Flask app yet.
# This allows the database to be configured and attached later when the app is created.
sqlalchemy_db = SQLAlchemy()

# Initialize the login manager extension (Flask-Login) without binding it to the app.
# This will manage user session handling, such as login and logout.
login_manager = LoginManager()

# host = 'localhost'
# port = 8786
#
# if check_socket(host, port):
#     dask_client = Client(":".join([host, str(port)]))
#     logger.info("Connected to existing Dask client")
# else:
#     cluster = LocalCluster(name='yapat_dask', n_workers=4, scheduler_port=port, dashboard_address=':8787')
#     dask_client = Client(cluster)
#     logger.info("Created new Dask client")
