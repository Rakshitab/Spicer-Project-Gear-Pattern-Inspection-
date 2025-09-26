from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class GearClassification(db.Model):
    """Model for storing gear image classification results"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(10), nullable=False)  # 'OK' or 'NOT OK'
    confidence = db.Column(db.Float, nullable=True)    # Confidence score (optional)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<GearClassification {self.filename}: {self.result}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'result': self.result,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }
