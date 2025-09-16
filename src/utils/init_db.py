"""Initialize the SQLite database with required tables."""
import os
from werkzeug.security import generate_password_hash
from src.schema_model import User, Dataset, EmbeddingMethod, EmbeddingResult
from src.extensions import sqlalchemy_db
from src.utils.settings import SQLALCHEMY_DATABASE_PATHS

def init_db(app, create_test_user=True):
    """Initialize all database tables and optionally create a test user."""
    # Create database directory if it doesn't exist
    for path in SQLALCHEMY_DATABASE_PATHS.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with app.app_context():
        # Create tables
        try:
            sqlalchemy_db.drop_all()
            sqlalchemy_db.create_all()

            if create_test_user:
                try:
                    # Create test user if it doesn't exist
                    test_user = User(
                        username='admin',
                        password=generate_password_hash('admin123')
                    )
                    sqlalchemy_db.session.add(test_user)
                    sqlalchemy_db.session.commit()
                except Exception as e:
                    print(f"Warning: Could not create test user: {e}")
                    sqlalchemy_db.session.rollback()
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

def reset_db(app):
    """Drop and recreate all tables (warning: this deletes all data)."""
    with app.app_context():
        sqlalchemy_db.drop_all()
        sqlalchemy_db.create_all()