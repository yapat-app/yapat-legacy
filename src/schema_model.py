from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship, backref

from src.extensions import sqlalchemy_db


# Define User model
class User(sqlalchemy_db.Model, UserMixin):
    __tablename__ = 'users'
    __bind_key__ = 'user_db'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password = Column(String(150), nullable=False)


# 1. Datasets Table
class Dataset(sqlalchemy_db.Model):
    __tablename__ = 'datasets'
    __bind_key__ = "pipeline_db"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(255), unique=True, nullable=False)
    path_audio = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_selected = Column(Boolean, default=False)  # True if selected, False otherwise



# 2. Embedding Methods Table
class EmbeddingMethod(sqlalchemy_db.Model):
    __tablename__ = 'embedding_methods'
    __bind_key__ = "pipeline_db"

    id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 3. Embeddings Results Table
class EmbeddingResult(sqlalchemy_db.Model):
    __tablename__ = 'embedding_results'
    __bind_key__ = "pipeline_db"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    embedding_id = Column(Integer, ForeignKey('embedding_methods.id'), nullable=False)
    file_path = Column(String(255), unique=True)  # Store path to the embedding file
    hyperparameters = Column(JSON, nullable=True)  # Hyperparameters stored as JSON
    evaluation_results = Column(JSON, nullable=True)
    task_state = Column(String(64), nullable=False)
    task_key = Column(String(64), unique=True)
    last_changed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset = relationship('Dataset', backref=backref('embedding_results', lazy=True))
    method = relationship('EmbeddingMethod', backref=backref('embedding_results', lazy=True))


# 4. Clustering Methods Table
class ClusteringMethod(sqlalchemy_db.Model):
    __tablename__ = 'clustering'
    __bind_key__ = "pipeline_db"

    id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 5. Clustering Results Table
class ClusteringResult(sqlalchemy_db.Model):
    __tablename__ = 'clustering_results'
    __bind_key__ = "pipeline_db"

    id = Column(Integer, primary_key=True)
    embedding_result_id = Column(Integer, ForeignKey('embedding_results.id'),
                          nullable=False)
    method_id = Column(Integer, ForeignKey('clustering.id'),
                       nullable=False)
    file_path = Column(String(255), unique=True)  # Store path to the clustering file
    hyperparameters = Column(JSON, nullable=True)  # Clustering hyperparameters stored as JSON
    evaluation_results = Column(JSON, nullable=True)
    task_state = Column(String(64), nullable=False)
    task_key = Column(String(64), unique=True)
    last_changed = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    embedding = relationship('EmbeddingResult', backref=backref('clustering_results', lazy=True))
    method = relationship('ClusteringMethod', backref=backref('clustering_results', lazy=True))


# 6. Dimensionality Reduction Methods Table
class DimReductionMethod(sqlalchemy_db.Model):
    __tablename__ = 'dimensionality_reduction'
    __bind_key__ = "pipeline_db"

    method_id = Column(Integer, primary_key=True)
    method_name = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# 7. Dimensionality Reduction Results Table
class DimReductionResult(sqlalchemy_db.Model):
    __tablename__ = 'dim_reduction_results'
    __bind_key__ = "pipeline_db"

    result_id = Column(Integer, primary_key=True)
    clustering_result_id = Column(Integer,
                                  ForeignKey('clustering_results.id'), nullable=False)
    method_id = Column(Integer, ForeignKey('dimensionality_reduction.method_id'),
                       nullable=False)
    reduction_file_path = Column(String(255),
                                 nullable=False)  # Path to the dimensionality reduction result file
    hyperparameters = Column(JSON,
                             nullable=True)  # Dimensionality reduction hyperparameters stored as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    clustering_result = relationship('ClusteringResult',
                                     backref=backref('dim_reduction_results', lazy=True))
    method = relationship('DimReductionMethod',
                          backref=backref('dim_reduction_results', lazy=True))
