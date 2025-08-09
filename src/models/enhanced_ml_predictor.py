import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.dataframe_utils import standardize_dataframe

# Initialize logger
logger = setup_logger('enhanced_ml_predictor')

class EnhancedMLPredictor:
    """
    Enhanced Machine Learning model for predicting price movements
    Uses ensemble methods and advanced feature engineering
    """
    
    def __init__(self, model_dir=None):
        """Initialize the ML predictor"""
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "data" / "models"
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def engineer_features(self, df):
        """Create advanced features for prediction"""
        df = df.copy()
        
        # Price momentum features
        if 'Close' in df.columns:
            df['Return_1D'] = df['Close'].pct_change(1)
            df['Return_5D'] = df['Close'].pct_change(5)
            df['Return_10D'] = df['Close'].pct_change(10)
        
            # Volatility features
            df['Volatility_5D'] = df['Return_1D'].rolling(5).std()
            df['Volatility_10D'] = df['Return_1D'].rolling(10).std()
        
        # Price patterns
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
            df['Price_Range_MA5'] = df['Price_Range'].rolling(5).mean()
        
        # RSI-related features
        if 'RSI' in df.columns:
            df['RSI_MA5'] = df['RSI'].rolling(5).mean()
            df['RSI_Change'] = df['RSI'] - df['RSI'].shift(1)
            
        # Moving average features
        if 'MA_Short' in df.columns and 'MA_Long' in df.columns:
            df['MA_Ratio'] = df['MA_Short'] / df['MA_Long']
            df['MA_Ratio_Change'] = df['MA_Ratio'] - df['MA_Ratio'].shift(1)
            
        # Volume-based features
        if 'Volume' in df.columns:
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
        
        # MACD features
        if 'MACD' in df.columns and 'Signal_Line' in df.columns:
            df['MACD_Diff'] = df['MACD'] - df['Signal_Line']
            df['MACD_Diff_Change'] = df['MACD_Diff'] - df['MACD_Diff'].shift(1)
        
        # Create target (1 if next day's close is higher)
        if 'Close' in df.columns:
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Remove features with high correlation
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        
        if high_corr_features:
            logger.info(f"Removing highly correlated features: {high_corr_features}")
            df.drop(high_corr_features, axis=1, inplace=True)
            
        return df
    
    def prepare_features(self, data):
        """Prepare data for model training/prediction"""
        # Handle MultiIndex if present
        df = standardize_dataframe(data)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Filter out non-feature columns
        exclude_cols = ['Target', 'Next_Day_Close', 'Buy_Signal', 'Sell_Signal']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Create X and y
        X = df[feature_cols] 
        y = df['Target']
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        return X, y
    
    def train_model(self, data, test_size=0.2, random_state=42, optimize=False):
        """Train an ensemble model"""
        logger.info("Preparing features for enhanced model training")
        
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize:
            logger.info("Performing hyperparameter tuning...")
            # Random Forest tuning
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(random_state=random_state)
            rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy')
            rf_grid.fit(X_train_scaled, y_train)
            best_rf = rf_grid.best_estimator_
            
            # Gradient Boosting tuning
            gb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
            gb = GradientBoostingClassifier(random_state=random_state)
            gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='accuracy')
            gb_grid.fit(X_train_scaled, y_train)
            best_gb = gb_grid.best_estimator_
            
            # Create voting classifier
            self.model = VotingClassifier(
                estimators=[
                    ('rf', best_rf),
                    ('gb', best_gb),
                    ('lr', LogisticRegression(random_state=random_state, max_iter=1000))
                ],
                voting='soft'
            )
            
            logger.info(f"Best RF params: {rf_grid.best_params_}")
            logger.info(f"Best GB params: {gb_grid.best_params_}")
        else:
            # Create a simpler ensemble if not optimizing
            self.model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=random_state)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
                    ('lr', LogisticRegression(random_state=random_state, max_iter=1000))
                ],
                voting='soft'
            )
        
        # Train the model
        logger.info("Training ensemble model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            # Get probabilities for ROC curve
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            logger.info(f"ROC AUC: {roc_auc:.4f}")
        except:
            logger.warning("Could not calculate ROC AUC")
        
        logger.info(f"Ensemble model training complete. Test accuracy: {accuracy:.4f}")
        logger.info("Classification report:\n" + classification_report(y_test, y_pred))
        
        # Feature importance if available (from Random Forest)
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.model.estimators_[0].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            logger.info("Top 10 important features:\n" + str(feature_importance.head(10)))
        
        # Save the model
        self.save_model("enhanced_price_predictor_model")
        
        return accuracy
    
    def predict(self, data):
        """Make predictions with the ensemble model"""
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        if not self.feature_names:
            logger.error("No feature names available")
            return None
            
        # Handle MultiIndex if present
        df = standardize_dataframe(data)
        
        # Engineer features
        try:
            df = self.engineer_features(df)
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Select only the features used in training that are available
        available_features = [f for f in self.feature_names if f in df.columns]
        
        if len(available_features) < len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
            
        if not available_features:
            logger.error("No usable features available")
            return None
        
        # Use only available features
        X = df[available_features]
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return None
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of class 1
            logger.info(f"Prediction probabilities (last 5): {probabilities[-5:]}")
        
        logger.info(f"Generated {len(predictions)} predictions: {predictions[-5:]} (last 5)")
        
        return predictions
        
    def save_model(self, filename="enhanced_price_predictor_model"):
        """Save the trained model and associated data"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        model_path = self.model_dir / f"{filename}.pkl"
        scaler_path = self.model_dir / f"{filename}_scaler.pkl"
        features_path = self.model_dir / f"{filename}_features.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        logger.info(f"Enhanced model saved to {model_path}")
    
    def load_model(self, filename="enhanced_price_predictor_model"):
        """Load a trained model"""
        model_path = self.model_dir / f"{filename}.pkl"
        scaler_path = self.model_dir / f"{filename}_scaler.pkl"
        features_path = self.model_dir / f"{filename}_features.pkl"
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            # Load feature names if available
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                    
            logger.info(f"Enhanced model loaded from {model_path}")
            logger.info(f"Using {len(self.feature_names)} features")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False