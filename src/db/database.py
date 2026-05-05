from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Optional, Dict
from config.settings import DB_PATH

Base = declarative_base()


class InferenceResultModel(Base):
    __tablename__ = 'inference_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    positive_prob = Column(Float, default=0.0)
    negative_prob = Column(Float, default=0.0)
    neutral_prob = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Float, default=0.0)
    true_label = Column(String(20), nullable=True)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'probabilities': {
                'positive': self.positive_prob,
                'negative': self.negative_prob,
                'neutral': self.neutral_prob
            },
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'processing_time_ms': self.processing_time_ms,
            'true_label': self.true_label
        }
    
    def is_correct(self) -> Optional[bool]:
        if self.true_label is None:
            return None
        return self.sentiment == self.true_label


class Database:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = DB_PATH
        
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        
        self._migrate()
        
        Session = sessionmaker(bind=self.engine)
        self.Session = Session
    
    def _migrate(self):
        from sqlalchemy import inspect, text
        inspector = inspect(self.engine)
        columns = [c['name'] for c in inspector.get_columns('inference_results')]
        
        if 'true_label' not in columns:
            with self.engine.connect() as conn:
                conn.execute(text("ALTER TABLE inference_results ADD COLUMN true_label VARCHAR(20)"))
                conn.commit()
    
    def add_result(
        self,
        text: str,
        sentiment: str,
        confidence: float,
        positive_prob: float = 0.0,
        negative_prob: float = 0.0,
        neutral_prob: float = 0.0,
        processing_time_ms: float = 0.0,
        true_label: str = None
    ) -> InferenceResultModel:
        session = self.Session()
        
        result = InferenceResultModel(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
            neutral_prob=neutral_prob,
            processing_time_ms=processing_time_ms,
            true_label=true_label
        )
        
        session.add(result)
        session.commit()
        session.refresh(result)
        
        session.close()
        
        return result
    
    def get_all_results(self, limit: int = None) -> List[InferenceResultModel]:
        session = self.Session()
        
        query = session.query(InferenceResultModel).order_by(
            InferenceResultModel.timestamp.desc()
        )
        
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        session.close()
        
        return results
    
    def get_results_by_sentiment(self, sentiment: str) -> List[InferenceResultModel]:
        session = self.Session()
        
        results = session.query(InferenceResultModel).filter(
            InferenceResultModel.sentiment == sentiment
        ).all()
        
        session.close()
        return results
    
    def get_sentiment_counts(self) -> Dict[str, int]:
        session = self.Session()
        
        results = session.query(InferenceResultModel).all()
        
        counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        for r in results:
            counts[r.sentiment] = counts.get(r.sentiment, 0) + 1
        
        session.close()
        return counts
    
    def get_daily_sentiment_counts(self, days: int = 30) -> List[Dict]:
        session = self.Session()
        
        results = session.query(InferenceResultModel).all()
        
        daily_counts = {}
        for r in results:
            date_str = r.timestamp.strftime('%Y-%m-%d')
            if date_str not in daily_counts:
                daily_counts[date_str] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
            daily_counts[date_str][r.sentiment] += 1
        
        session.close()
        
        return [
            {'date': date, **counts}
            for date, counts in sorted(daily_counts.items())
        ]
    
    def get_negative_reviews(self, limit: int = None) -> List[InferenceResultModel]:
        session = self.Session()
        
        query = session.query(InferenceResultModel).filter(
            InferenceResultModel.sentiment == 'Negative'
        ).order_by(InferenceResultModel.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        results = query.all()
        session.close()
        
        return results
    
    def get_incorrect_predictions(self, limit: int = 3) -> List[InferenceResultModel]:
        session = self.Session()
        
        results = session.query(InferenceResultModel).filter(
            InferenceResultModel.true_label != None,
            InferenceResultModel.true_label != InferenceResultModel.sentiment
        ).order_by(InferenceResultModel.confidence.asc()).limit(limit).all()
        
        session.close()
        return results
    
    def get_statistics(self) -> Dict:
        session = self.Session()
        
        total = session.query(InferenceResultModel).count()
        sentiment_counts = self.get_sentiment_counts()
        
        avg_confidence = session.query(
            func.avg(InferenceResultModel.confidence)
        ).scalar() or 0
        
        avg_processing_time = session.query(
            func.avg(InferenceResultModel.processing_time_ms)
        ).scalar() or 0
        
        session.close()
        
        return {
            'total_inferences': total,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': float(avg_confidence),
            'average_processing_time_ms': float(avg_processing_time)
        }
    
    def clear_all(self):
        session = self.Session()
        session.query(InferenceResultModel).delete()
        session.commit()
        session.close()
