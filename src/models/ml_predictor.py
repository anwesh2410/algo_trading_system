import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from config.config_loader import Config

# Initialize logger
logger = setup_logger('ml_predictor')

class MLPredictor:
    """
    Machine Learning model for predicting price movements
    """
    
    def __init__(self, model_dir=None):
        """Initialize the ML predictor"""
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "data" / "models"
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []  # Store feature names used during training
        
    def prepare_features(self, data):
        """
        Prepare features for the ML model
        
        Args:
            data (DataFrame): Stock data with technical indicators
            
        Returns:
            tuple: X (features), y (target)
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Debug the column structure
        logger.info(f"DataFrame column structure: {type(df.columns)}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # For MultiIndex columns, let's check the levels
        if isinstance(df.columns, pd.MultiIndex):
            logger.info(f"MultiIndex levels: {df.columns.levels}")
            logger.info(f"First few column pairs: {list(df.columns[:5])}")
            
            # Extract the price data and indicators
            simple_df = df.droplevel(1, axis=1)
            
            # Log the simplified columns
            logger.info(f"Simplified columns: {list(simple_df.columns)}")
            
            # Use the simplified DataFrame
            df = simple_df
        
        # Check if expected columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 
                        'MA_Short', 'MA_Long', 'MACD']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            
            # Attempt to find columns with similar names
            for col in missing_cols:
                candidates = [c for c in df.columns if col.lower() in c.lower()]
                if candidates:
                    logger.info(f"Found potential match for {col}: {candidates}")
                    # Use the first match
                    df[col] = df[candidates[0]]
        
        # Create target variable - 1 if tomorrow's close is higher than today's, 0 otherwise
        if 'Close' in df.columns:
            df['Next_Day_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Next_Day_Close'] > df['Close']).astype(int)
            
            # Drop NaN values created by the shift
            df = df.dropna()
            
            # Create basic features if they exist
            features = []
            
            # Add technical indicators if available
            for indicator in ['RSI', 'MA_Short', 'MA_Long', 'MACD', 'MACD_Histogram', 
                              'Signal_Line', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_MA']:
                if indicator in df.columns:
                    features.append(indicator)
            
            # Add price-based features
            if 'Close' in df.columns and 'Volume' in df.columns:
                df['Close_Pct_Change'] = df['Close'].pct_change()
                features.append('Close_Pct_Change')
                
                df['Volume_Pct_Change'] = df['Volume'].pct_change()
                features.append('Volume_Pct_Change')
            
            if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
                df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
                features.append('High_Low_Diff')
            
            if 'Close' in df.columns and 'MA_Short' in df.columns:
                df['Close_MA_Short_Ratio'] = df['Close'] / df['MA_Short']
                features.append('Close_MA_Short_Ratio')
            
            # Drop any rows with NaN values
            df = df.dropna()
            
            # Extract features and target
            X = df[features]
            y = df['Target']
            
            logger.info(f"Prepared {len(X)} samples with {len(features)} features")
            
            return X, y
        else:
            logger.error("Close price column not found, cannot prepare features")
            raise KeyError("Close price column not found in DataFrame")
    
    def train_model(self, data, test_size=0.2, random_state=42):
        """
        Train a Random Forest model on the data
        
        Args:
            data (DataFrame): Stock data with technical indicators
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            float: Model accuracy on test set
        """
        logger.info("Preparing features for model training")
        X, y = self.prepare_features(data)
        
        # Store feature names for prediction later
        self.feature_names = list(X.columns)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        logger.info("Training Random Forest model")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Test accuracy: {accuracy:.4f}")
        logger.info("Classification report:\n" + classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        logger.info("Top 5 important features:\n" + str(feature_importance.head()))
        
        # Save the model and scaler
        self.save_model()
        
        return accuracy
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data (DataFrame): Stock data with technical indicators
            
        Returns:
            array: Predicted values (0 for down, 1 for up)
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        # Check if feature names were saved during training
        if not self.feature_names:
            logger.error("No feature names available. Model may not have been trained properly.")
            return None
            
        logger.info(f"Using these features for prediction: {self.feature_names}")
        
        # Prepare data the same way as in training
        df = data.copy().iloc[:-1]  # Remove the last row as we don't have Next_Day_Close
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            simple_df = df.droplevel(1, axis=1)
            df = simple_df
        
        # Create the same features as during training
        features_to_use = []
        
        # Add technical indicators if available
        for indicator in ['RSI', 'MA_Short', 'MA_Long', 'MACD', 'MACD_Histogram', 
                          'Signal_Line', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_MA']:
            if indicator in df.columns and indicator in self.feature_names:
                features_to_use.append(indicator)
        
        # Add price-based features
        if 'Close' in df.columns and 'Volume' in df.columns:
            if 'Close_Pct_Change' in self.feature_names:
                df['Close_Pct_Change'] = df['Close'].pct_change()
                features_to_use.append('Close_Pct_Change')
            
            if 'Volume_Pct_Change' in self.feature_names:
                df['Volume_Pct_Change'] = df['Volume'].pct_change()
                features_to_use.append('Volume_Pct_Change')
        
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            if 'High_Low_Diff' in self.feature_names:
                df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
                features_to_use.append('High_Low_Diff')
        
        if 'Close' in df.columns and 'MA_Short' in df.columns:
            if 'Close_MA_Short_Ratio' in self.feature_names:
                df['Close_MA_Short_Ratio'] = df['Close'] / df['MA_Short']
                features_to_use.append('Close_MA_Short_Ratio')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Check if all required features are available
        missing_features = [f for f in self.feature_names if f not in features_to_use]
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}")
            
        # Use only the available features that were used in training
        X = df[features_to_use]
        
        # Log the shape of the prediction data
        logger.info(f"Prediction data shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        logger.info(f"Generated {len(predictions)} predictions: {predictions[-5:]} (last 5)")
        
        return predictions
    
    def save_model(self, filename=None):
        """
        Save the trained model, scaler, and feature names to disk
        
        Args:
            filename (str): Optional custom filename
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        if filename is None:
            filename = "price_predictor_model"
        
        model_path = self.model_dir / f"{filename}.pkl"
        scaler_path = self.model_dir / f"{filename}_scaler.pkl"
        features_path = self.model_dir / f"{filename}_features.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, filename=None):
        """
        Load a trained model from disk
        
        Args:
            filename (str): Optional custom filename
        
        Returns:
            bool: True if successful, False otherwise
        """
        if filename is None:
            filename = "price_predictor_model"
        
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
                    
            logger.info(f"Model loaded from {model_path} with features: {self.feature_names}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False