import logging

from sqlalchemy.exc import SQLAlchemyError

from src.extensions import sqlalchemy_db
from src.schema_model import Dataset
from src.utils import server

logger = logging.getLogger(__name__)


def register_dataset(dataset_name, path_audio, flask_server=server):
    with flask_server.app_context():
        try:
            sqlalchemy_db.session.add(Dataset(dataset_name=dataset_name, path_audio=path_audio))
            sqlalchemy_db.session.commit()
        except SQLAlchemyError as e:
            sqlalchemy_db.session.rollback()
            logger.exception(e)
