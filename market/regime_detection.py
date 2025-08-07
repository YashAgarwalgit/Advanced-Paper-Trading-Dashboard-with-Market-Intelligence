"""
Advanced market regime detection algorithms with institutional-grade analytics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.stats import norm, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm
import warnings
from config import Config
from utils.helpers import safe_float, get_timestamp_iso
from utils.decorators import measure_performance

warnings.filterwarnings("ignore")

class RegimeState(Enum):
    """Market regime states"""
    BULL_MARKET = "bull_market"          # Strong uptrend, low volatility
    BEAR_MARKET = "bear_market"          # Strong downtrend, high volatility  
    VOLATILE_BULL = "volatile_bull"      # Uptrend with high volatility
    VOLATILE_BEAR = "volatile_bear"      # Downtrend with high volatility
    SIDEWAYS_LOW_VOL = "sideways_low"    # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high"  # Range-bound, high volatility
    CRISIS = "crisis"                    # Extreme volatility, market stress
    RECOVERY = "recovery"                # Post-crisis recovery phase

class RegimeIndicator(Enum):
    """Regime detection indicators"""
    PRICE_MOMENTUM = "price_momentum"
    VOLATILITY_LEVEL = "volatility_level"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    TREND_STRENGTH = "trend_strength"
    MARKET_BREADTH = "market_breadth"
    VOLUME_ANALYSIS = "volume_analysis"
    CORRELATION_STRUCTURE = "correlation_structure"
    RISK_PREMIUM = "risk_premium"

@dataclass
class RegimeTransition:
    """Regime transition data"""
    from_state: RegimeState
    to_state: RegimeState
    transition_date: datetime
    probability: float
    duration_days: int
    trigger_factors: List[str]
    confidence: float

@dataclass
class RegimeDetectionResult:
    """Regime detection result"""
    current_state: RegimeState
    confidence_score: float
    state_probabilities: Dict[RegimeState, float]
    regime_indicators: Dict[RegimeIndicator, float]
    detection_timestamp: datetime
    expected_duration: Optional[int] = None
    transition_signals: Optional[List[str]] = None

@dataclass
class RegimeStatistics:
    """Regime statistical properties"""
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    max_drawdown: float
    var_95: float
    duration_stats: Dict[str, float]
    transition_matrix: np.ndarray

class MarketRegimeDetector:
    """
    Advanced market regime detection system using multiple methodologies:
    
    1. Hidden Markov Models (HMM) for regime identification
    2. Gaussian Mixture Models for regime clustering
    3. Volatility-based regime detection
    4. Momentum and trend-based classification
    5. Multi-dimensional regime analysis
    6. Real-time regime monitoring
    7. Regime transition prediction
    """
    
    def __init__(self, n_regimes: int = 4, lookback_window: int = 252):
        self.logger = logging.getLogger(__name__)
        self.logger.info("MarketRegimeDetector.__init__ ENTRY")
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.logger = logging.getLogger(__name__)
        import sys
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        print('[regime_detection.py][__init__] Logger initialized')
        self.logger.info("MarketRegimeDetector.__init__ EXIT")
        
        # Model parameters
        self.hmm_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        # Regime thresholds
        self.volatility_thresholds = {
            'low': 0.15,    # 15% annualized
            'medium': 0.25, # 25% annualized
            'high': 0.40    # 40% annualized
        }
        
        self.momentum_thresholds = {
            'strong_bull': 0.20,   # 20% momentum
            'weak_bull': 0.05,     # 5% momentum
            'weak_bear': -0.05,    # -5% momentum
            'strong_bear': -0.20   # -20% momentum
        }
        
        # Historical regime data
        self.regime_history: List[RegimeDetectionResult] = []
        self.transition_history: List[RegimeTransition] = []
        
        # Regime statistics
        self.regime_stats: Dict[RegimeState, RegimeStatistics] = {}
    
    @measure_performance
    async def detect_current_regime(self, market_data: Dict[str, Any]) -> RegimeDetectionResult:
        self.logger.info("MarketRegimeDetector.detect_current_regime ENTRY")
        """
        Detect current market regime using ensemble of methods
        
        Args:
            market_data: Comprehensive market data including prices, volumes, volatility
            
        Returns:
            RegimeDetectionResult with current state and confidence
        """
        
        try:
            self.logger.info("Starting comprehensive regime detection")
            
            # Prepare feature matrix
            features = await self._prepare_regime_features(market_data)
            
            if features is None or len(features) < 10:
                return self._get_default_regime_result()
            
            # Apply multiple detection methods
            detection_results = await self._apply_ensemble_detection(features)
            
            # Calculate regime indicators
            regime_indicators = await self._calculate_regime_indicators(market_data, features)
            
            # Determine final regime state
            final_state, confidence = self._consensus_regime_determination(detection_results)
            
            # Calculate state probabilities
            state_probabilities = self._calculate_state_probabilities(detection_results)
            
            # Detect transition signals
            transition_signals = await self._detect_transition_signals(features, final_state)
            
            # Estimate regime duration
            expected_duration = await self._estimate_regime_duration(final_state, features)
            
            result = RegimeDetectionResult(
                current_state=final_state,
                confidence_score=confidence,
                state_probabilities=state_probabilities,
                regime_indicators=regime_indicators,
                detection_timestamp=datetime.utcnow(),
                expected_duration=expected_duration,
                transition_signals=transition_signals
            )
            
            # Update history
            self.regime_history.append(result)
            await self._check_regime_transitions(result)
            
            self.logger.info(f"Regime detection completed: {final_state.value} (confidence: {confidence:.2f})")
            self.logger.info("MarketRegimeDetector.detect_current_regime EXIT (success)")
            return result
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return self._get_default_regime_result()
    
    async def _prepare_regime_features(self, market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare comprehensive feature matrix for regime detection"""
        
        try:
            features_list = []
            
            # Extract price-based features
            if 'price_data' in market_data:
                price_features = self._extract_price_features(market_data['price_data'])
                features_list.append(price_features)
            
            # Extract volatility features
            if 'volatility_data' in market_data:
                vol_features = self._extract_volatility_features(market_data['volatility_data'])
                features_list.append(vol_features)
            
            # Extract volume features
            if 'volume_data' in market_data:
                volume_features = self._extract_volume_features(market_data['volume_data'])
                features_list.append(volume_features)
            
            # Extract cross-asset features
            if 'cross_asset_data' in market_data:
                cross_features = self._extract_cross_asset_features(market_data['cross_asset_data'])
                features_list.append(cross_features)
            
            # Extract macro features
            if 'macro_data' in market_data:
                macro_features = self._extract_macro_features(market_data['macro_data'])
                features_list.append(macro_features)
            
            # Combine all features
            if features_list:
                combined_features = pd.concat(features_list, axis=1)
                
                # Handle missing values
                combined_features = combined_features.fillna(method='ffill').fillna(0)
                
                # Ensure minimum data length
                if len(combined_features) >= 20:
                    return combined_features.tail(self.lookback_window)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _extract_price_features(self, price_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Extract price-based regime features"""
        
        try:
            if isinstance(price_data, dict):
                # Convert dict to DataFrame
                df = pd.DataFrame(price_data)
            else:
                df = price_data.copy()
            
            if 'Close' not in df.columns:
                df['Close'] = df.iloc[:, 0]  # Use first column as close price
            
            features = pd.DataFrame(index=df.index)
            
            # Returns
            features['returns'] = df['Close'].pct_change()
            features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages and trends
            features['ma_5'] = df['Close'].rolling(5).mean()
            features['ma_20'] = df['Close'].rolling(20).mean()
            features['ma_50'] = df['Close'].rolling(50).mean()
            
            # Price relative to moving averages
            features['price_ma5_ratio'] = df['Close'] / features['ma_5']
            features['price_ma20_ratio'] = df['Close'] / features['ma_20']
            features['price_ma50_ratio'] = df['Close'] / features['ma_50']
            
            # Momentum indicators
            features['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            features['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            features['momentum_60'] = df['Close'] / df['Close'].shift(60) - 1
            
            # Price position in range
            features['high_20'] = df['Close'].rolling(20).max()
            features['low_20'] = df['Close'].rolling(20).min()
            features['price_position'] = (df['Close'] - features['low_20']) / (features['high_20'] - features['low_20'])
            
            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength_series(df['Close'])
            
            return features.dropna()
            
        except Exception as e:
            self.logger.warning(f"Price feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_volatility_features(self, volatility_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Extract volatility-based regime features"""
        
        try:
            if isinstance(volatility_data, dict):
                df = pd.DataFrame(volatility_data)
            else:
                df = volatility_data.copy()
            
            if 'Close' not in df.columns:
                df['Close'] = df.iloc[:, 0]
            
            features = pd.DataFrame(index=df.index)
            
            # Calculate returns for volatility measures
            returns = df['Close'].pct_change().dropna()
            
            # Rolling volatilities
            features['volatility_5'] = returns.rolling(5).std() * np.sqrt(252)
            features['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
            features['volatility_60'] = returns.rolling(60).std() * np.sqrt(252)
            
            # Volatility of volatility
            features['vol_of_vol'] = features['volatility_20'].rolling(20).std()
            
            # GARCH-like features
            features['squared_returns'] = returns ** 2
            features['garch_vol'] = features['squared_returns'].ewm(alpha=0.1).mean()
            
            # Volatility clustering
            features['vol_clustering'] = self._detect_volatility_clustering_series(returns)
            
            # Realized vs implied volatility (simplified)
            features['realized_vol'] = returns.rolling(20).std() * np.sqrt(252)
            
            # Volatility percentiles
            features['vol_percentile'] = features['volatility_20'].rolling(252).rank(pct=True)
            
            return features.dropna()
            
        except Exception as e:
            self.logger.warning(f"Volatility feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_volume_features(self, volume_data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """Extract volume-based regime features"""
        
        try:
            if isinstance(volume_data, dict):
                df = pd.DataFrame(volume_data)
            else:
                df = volume_data.copy()
            
            features = pd.DataFrame(index=df.index)
            
            if 'Volume' in df.columns:
                # Volume moving averages
                features['volume_ma_20'] = df['Volume'].rolling(20).mean()
                features['volume_ratio'] = df['Volume'] / features['volume_ma_20']
                
                # Volume momentum
                features['volume_momentum'] = df['Volume'] / df['Volume'].shift(20) - 1
                
                # Volume volatility
                features['volume_volatility'] = df['Volume'].rolling(20).std() / features['volume_ma_20']
                
                # Volume percentile
                features['volume_percentile'] = df['Volume'].rolling(252).rank(pct=True)
            
            if 'Close' in df.columns and 'Volume' in df.columns:
                # Price-volume relationship
                price_change = df['Close'].pct_change()
                volume_change = df['Volume'].pct_change()
                features['price_volume_corr'] = price_change.rolling(20).corr(volume_change)
                
                # On-balance volume (simplified)
                obv = (price_change.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) * df['Volume']).cumsum()
                features['obv_momentum'] = obv / obv.shift(20) - 1
            
            return features.dropna()
            
        except Exception as e:
            self.logger.warning(f"Volume feature extraction failed: {e}")
            return pd.DataFrame()
    
    async def _apply_ensemble_detection(self, features: pd.DataFrame) -> Dict[str, RegimeState]:
        """Apply ensemble of regime detection methods"""
        
        try:
            detection_results = {}
            
            # Method 1: Hidden Markov Model
            hmm_result = await self._hmm_regime_detection(features)
            detection_results['hmm'] = hmm_result
            
            # Method 2: Gaussian Mixture Model
            gmm_result = await self._gmm_regime_detection(features)
            detection_results['gmm'] = gmm_result
            
            # Method 3: Volatility-based classification
            vol_result = await self._volatility_regime_detection(features)
            detection_results['volatility'] = vol_result
            
            # Method 4: Momentum-based classification
            momentum_result = await self._momentum_regime_detection(features)
            detection_results['momentum'] = momentum_result
            
            # Method 5: PCA-based clustering
            pca_result = await self._pca_regime_detection(features)
            detection_results['pca'] = pca_result
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Ensemble detection failed: {e}")
            return {'default': RegimeState.SIDEWAYS_LOW_VOL}
    
    async def _hmm_regime_detection(self, features: pd.DataFrame) -> RegimeState:
        """Hidden Markov Model regime detection"""
        
        try:
            # Prepare features for HMM
            feature_cols = ['returns', 'volatility_20', 'momentum_20', 'trend_strength']
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if len(available_cols) < 2:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            X = features[available_cols].values
            X = self.scaler.fit_transform(X)
            
            # Fit HMM model
            model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full", random_state=42)
            model.fit(X)
            
            # Predict current state
            states = model.predict(X)
            current_state_idx = states[-1]
            
            # Map to regime state
            regime_mapping = self._create_hmm_regime_mapping(model, X)
            current_regime = regime_mapping.get(current_state_idx, RegimeState.SIDEWAYS_LOW_VOL)
            
            self.hmm_model = model
            return current_regime
            
        except Exception as e:
            self.logger.warning(f"HMM regime detection failed: {e}")
            return RegimeState.SIDEWAYS_LOW_VOL
    
    async def _gmm_regime_detection(self, features: pd.DataFrame) -> RegimeState:
        """Gaussian Mixture Model regime detection"""
        
        try:
            # Select features for GMM
            feature_cols = ['returns', 'volatility_20', 'momentum_5', 'momentum_20']
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if len(available_cols) < 2:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            X = features[available_cols].dropna().values
            X = self.scaler.fit_transform(X)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
            gmm.fit(X)
            
            # Predict current cluster
            current_obs = X[-1].reshape(1, -1)
            cluster = gmm.predict(current_obs)[0]
            
            # Map cluster to regime
            regime_mapping = self._create_gmm_regime_mapping(gmm, X)
            current_regime = regime_mapping.get(cluster, RegimeState.SIDEWAYS_LOW_VOL)
            
            self.gmm_model = gmm
            return current_regime
            
        except Exception as e:
            self.logger.warning(f"GMM regime detection failed: {e}")
            return RegimeState.SIDEWAYS_LOW_VOL
    
    async def _volatility_regime_detection(self, features: pd.DataFrame) -> RegimeState:
        """Volatility-based regime classification"""
        
        try:
            if 'volatility_20' not in features.columns:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            current_vol = features['volatility_20'].iloc[-1]
            vol_trend = features['volatility_20'].diff(5).iloc[-1] if len(features) > 5 else 0
            
            # Classify based on volatility level and trend
            if current_vol > self.volatility_thresholds['high']:
                return RegimeState.CRISIS
            elif current_vol > self.volatility_thresholds['medium']:
                if vol_trend > 0.05:
                    return RegimeState.VOLATILE_BEAR
                else:
                    return RegimeState.VOLATILE_BULL
            elif current_vol < self.volatility_thresholds['low']:
                return RegimeState.SIDEWAYS_LOW_VOL
            else:
                return RegimeState.SIDEWAYS_HIGH_VOL
                
        except Exception as e:
            self.logger.warning(f"Volatility regime detection failed: {e}")
            return RegimeState.SIDEWAYS_LOW_VOL
    
    async def _momentum_regime_detection(self, features: pd.DataFrame) -> RegimeState:
        """Momentum-based regime classification"""
        
        try:
            if 'momentum_20' not in features.columns:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            current_momentum = features['momentum_20'].iloc[-1]
            vol_level = features.get('volatility_20', pd.Series([0.2])).iloc[-1]
            
            # Classify based on momentum and volatility
            if current_momentum > self.momentum_thresholds['strong_bull']:
                return RegimeState.VOLATILE_BULL if vol_level > 0.25 else RegimeState.BULL_MARKET
            elif current_momentum > self.momentum_thresholds['weak_bull']:
                return RegimeState.BULL_MARKET
            elif current_momentum < self.momentum_thresholds['strong_bear']:
                return RegimeState.VOLATILE_BEAR if vol_level > 0.25 else RegimeState.BEAR_MARKET
            elif current_momentum < self.momentum_thresholds['weak_bear']:
                return RegimeState.BEAR_MARKET
            else:
                return RegimeState.SIDEWAYS_HIGH_VOL if vol_level > 0.25 else RegimeState.SIDEWAYS_LOW_VOL
                
        except Exception as e:
            self.logger.warning(f"Momentum regime detection failed: {e}")
            return RegimeState.SIDEWAYS_LOW_VOL
    
    def _consensus_regime_determination(self, detection_results: Dict[str, RegimeState]) -> Tuple[RegimeState, float]:
        """Determine consensus regime from multiple methods"""
        
        try:
            # Count votes for each regime
            regime_votes = {}
            for method, regime in detection_results.items():
                regime_votes[regime] = regime_votes.get(regime, 0) + 1
            
            # Find most voted regime
            if regime_votes:
                consensus_regime = max(regime_votes, key=regime_votes.get)
                confidence = regime_votes[consensus_regime] / len(detection_results)
                return consensus_regime, confidence
            
            return RegimeState.SIDEWAYS_LOW_VOL, 0.5
            
        except Exception:
            return RegimeState.SIDEWAYS_LOW_VOL, 0.5
    
    async def calculate_transition_probability(self, market_data: Dict[str, Any]) -> float:
        """Calculate regime transition probability"""
        
        try:
            # This would implement sophisticated transition probability calculation
            # using historical transition patterns, current regime duration, etc.
            
            # Simplified implementation
            if len(self.regime_history) < 2:
                return 0.1  # Low transition probability without history
            
            current_regime = self.regime_history[-1].current_state
            regime_duration = len([r for r in self.regime_history[-10:] if r.current_state == current_regime])
            
            # Higher probability if regime has lasted long
            base_probability = min(0.8, regime_duration * 0.1)
            
            # Adjust based on market stress
            if 'market_stress' in market_data:
                stress_level = market_data['market_stress']
                base_probability += stress_level * 0.3
            
            return min(0.9, base_probability)
            
        except Exception:
            return 0.1
    
    async def estimate_regime_duration(self, regime_state: RegimeState) -> int:
        """Estimate expected regime duration in days"""
        
        try:
            # Historical average durations (simplified)
            duration_estimates = {
                RegimeState.BULL_MARKET: 180,
                RegimeState.BEAR_MARKET: 120,
                RegimeState.VOLATILE_BULL: 60,
                RegimeState.VOLATILE_BEAR: 45,
                RegimeState.SIDEWAYS_LOW_VOL: 90,
                RegimeState.SIDEWAYS_HIGH_VOL: 30,
                RegimeState.CRISIS: 15,
                RegimeState.RECOVERY: 75
            }
            
            return duration_estimates.get(regime_state, 60)
            
        except Exception:
            return 60
    
    def _get_default_regime_result(self) -> RegimeDetectionResult:
        """Get default regime result when detection fails"""
        
        return RegimeDetectionResult(
            current_state=RegimeState.SIDEWAYS_LOW_VOL,
            confidence_score=0.5,
            state_probabilities={state: 1.0/len(RegimeState) for state in RegimeState},
            regime_indicators={indicator: 0.5 for indicator in RegimeIndicator},
            detection_timestamp=datetime.utcnow(),
            expected_duration=60,
            transition_signals=[]
        )
    
    async def _calculate_regime_indicators(self, market_data: Dict, features: pd.DataFrame) -> Dict[RegimeIndicator, float]:
        """Calculate individual regime indicators"""
        
        try:
            indicators = {}
            
            # Price momentum indicator
            if 'momentum_20' in features.columns:
                momentum = features['momentum_20'].iloc[-1]
                indicators[RegimeIndicator.PRICE_MOMENTUM] = max(0, min(10, momentum * 50 + 5))
            
            # Volatility level indicator
            if 'volatility_20' in features.columns:
                vol = features['volatility_20'].iloc[-1]
                indicators[RegimeIndicator.VOLATILITY_LEVEL] = max(0, min(10, vol * 25))
            
            # Volatility clustering indicator
            clustering_detected = self._detect_volatility_clustering_series(features.get('returns', pd.Series()))
            indicators[RegimeIndicator.VOLATILITY_CLUSTERING] = 7.0 if clustering_detected else 3.0
            
            # Trend strength indicator
            if 'trend_strength' in features.columns:
                trend = features['trend_strength'].iloc[-1]
                indicators[RegimeIndicator.TREND_STRENGTH] = max(0, min(10, trend / 10))
            
            # Market breadth (mock)
            indicators[RegimeIndicator.MARKET_BREADTH] = market_data.get('breadth_ratio', 0.5) * 10
            
            # Volume analysis (mock)
            indicators[RegimeIndicator.VOLUME_ANALYSIS] = market_data.get('volume_sentiment', 50) / 10
            
            # Correlation structure
            avg_correlation = market_data.get('average_correlation', 0.5)
            indicators[RegimeIndicator.CORRELATION_STRUCTURE] = avg_correlation * 10
            
            # Risk premium (mock)
            indicators[RegimeIndicator.RISK_PREMIUM] = 5.0  # Neutral default
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Regime indicators calculation failed: {e}")
            return {indicator: 5.0 for indicator in RegimeIndicator}

    def _calculate_state_probabilities(self, detection_results: Dict[str, RegimeState]) -> Dict[RegimeState, float]:
        """Calculate probability distribution over regime states"""
        
        try:
            # Count votes for each state
            state_votes = {}
            total_methods = len(detection_results)
            
            for state in RegimeState:
                state_votes[state] = 0
            
            for method, state in detection_results.items():
                state_votes[state] += 1
            
            # Convert to probabilities
            state_probabilities = {}
            for state, votes in state_votes.items():
                state_probabilities[state] = votes / total_methods if total_methods > 0 else 1.0 / len(RegimeState)
            
            return state_probabilities
            
        except Exception as e:
            self.logger.error(f"State probability calculation failed: {e}")
            return {state: 1.0 / len(RegimeState) for state in RegimeState}

    async def _detect_transition_signals(self, features: pd.DataFrame, current_state: RegimeState) -> List[str]:
        """Detect potential regime transition signals"""
        
        try:
            signals = []
            
            if len(features) < 10:
                return signals
            
            # Volatility regime transition signals
            if 'volatility_20' in features.columns:
                recent_vol = features['volatility_20'].iloc[-5:].mean()
                historical_vol = features['volatility_20'].iloc[:-5].mean()
                
                if recent_vol > historical_vol * 1.5:
                    signals.append("Volatility breakout - potential regime shift to higher volatility state")
                elif recent_vol < historical_vol * 0.7:
                    signals.append("Volatility compression - potential regime shift to lower volatility state")
            
            # Momentum regime transition signals
            if 'momentum_20' in features.columns:
                recent_momentum = features['momentum_20'].iloc[-3:].mean()
                
                if current_state in [RegimeState.BEAR_MARKET, RegimeState.VOLATILE_BEAR] and recent_momentum > 0.05:
                    signals.append("Positive momentum emerging in bear regime - potential bullish transition")
                elif current_state in [RegimeState.BULL_MARKET, RegimeState.VOLATILE_BULL] and recent_momentum < -0.05:
                    signals.append("Negative momentum in bull regime - potential bearish transition")
            
            # Trend strength transitions
            if 'trend_strength' in features.columns:
                trend_change = features['trend_strength'].diff().iloc[-5:].mean()
                
                if trend_change > 10:
                    signals.append("Strengthening trend - regime consolidation expected")
                elif trend_change < -10:
                    signals.append("Weakening trend - potential regime transition ahead")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Transition signal detection failed: {e}")
            return ["Transition analysis temporarily unavailable"]

    async def _check_regime_transitions(self, current_result: RegimeDetectionResult):
        """Check for regime transitions and update history"""
        
        try:
            if len(self.regime_history) < 2:
                return  # Need history to detect transitions
            
            previous_result = self.regime_history[-2]
            
            # Check if regime changed
            if previous_result.current_state != current_result.current_state:
                # Create transition record
                transition = RegimeTransition(
                    from_state=previous_result.current_state,
                    to_state=current_result.current_state,
                    transition_date=current_result.detection_timestamp,
                    probability=current_result.confidence_score,
                    duration_days=(current_result.detection_timestamp - previous_result.detection_timestamp).days,
                    trigger_factors=current_result.transition_signals or [],
                    confidence=current_result.confidence_score
                )
                
                self.transition_history.append(transition)
                
                self.logger.info(f"Regime transition detected: {transition.from_state.value} -> {transition.to_state.value}")
            
        except Exception as e:
            self.logger.error(f"Regime transition check failed: {e}")

    def _create_hmm_regime_mapping(self, model, data: np.ndarray) -> Dict[int, RegimeState]:
        """Create mapping from HMM states to regime states"""
        
        try:
            n_states = model.n_components
            state_mapping = {}
            
            # Decode states
            states = model.predict(data)
            
            # Analyze each state's characteristics
            for state_idx in range(n_states):
                state_mask = states == state_idx
                
                if np.sum(state_mask) == 0:
                    state_mapping[state_idx] = RegimeState.SIDEWAYS_LOW_VOL
                    continue
                
                # Get state data
                state_data = data[state_mask]
                
                if state_data.shape[1] >= 2:  # At least returns and volatility
                    avg_return = np.mean(state_data[:, 0])  # Assuming first column is returns
                    avg_volatility = np.mean(state_data[:, 1])  # Assuming second column is volatility
                    
                    # Map based on return and volatility characteristics
                    if avg_return > 0.01 and avg_volatility < 0.02:
                        state_mapping[state_idx] = RegimeState.BULL_MARKET
                    elif avg_return > 0.01 and avg_volatility >= 0.02:
                        state_mapping[state_idx] = RegimeState.VOLATILE_BULL
                    elif avg_return < -0.01 and avg_volatility < 0.02:
                        state_mapping[state_idx] = RegimeState.BEAR_MARKET
                    elif avg_return < -0.01 and avg_volatility >= 0.02:
                        state_mapping[state_idx] = RegimeState.VOLATILE_BEAR
                    elif avg_volatility > 0.03:
                        state_mapping[state_idx] = RegimeState.CRISIS
                    elif avg_volatility < 0.015:
                        state_mapping[state_idx] = RegimeState.SIDEWAYS_LOW_VOL
                    else:
                        state_mapping[state_idx] = RegimeState.SIDEWAYS_HIGH_VOL
                else:
                    state_mapping[state_idx] = RegimeState.SIDEWAYS_LOW_VOL
            
            return state_mapping
            
        except Exception as e:
            self.logger.error(f"HMM regime mapping failed: {e}")
            return {i: RegimeState.SIDEWAYS_LOW_VOL for i in range(model.n_components)}

    def _create_gmm_regime_mapping(self, model, data: np.ndarray) -> Dict[int, RegimeState]:
        """Create mapping from GMM clusters to regime states"""
        
        try:
            n_clusters = model.n_components
            cluster_mapping = {}
            
            # Get cluster assignments
            clusters = model.predict(data)
            
            # Analyze each cluster's characteristics
            for cluster_idx in range(n_clusters):
                cluster_mask = clusters == cluster_idx
                
                if np.sum(cluster_mask) == 0:
                    cluster_mapping[cluster_idx] = RegimeState.SIDEWAYS_LOW_VOL
                    continue
                
                cluster_data = data[cluster_mask]
                
                if cluster_data.shape[1] >= 2:
                    avg_return = np.mean(cluster_data[:, 0])
                    avg_volatility = np.mean(cluster_data[:, 1])
                    
                    # Similar mapping logic as HMM
                    if avg_return > 0.01 and avg_volatility < 0.02:
                        cluster_mapping[cluster_idx] = RegimeState.BULL_MARKET
                    elif avg_return > 0.01 and avg_volatility >= 0.02:
                        cluster_mapping[cluster_idx] = RegimeState.VOLATILE_BULL
                    elif avg_return < -0.01 and avg_volatility < 0.02:
                        cluster_mapping[cluster_idx] = RegimeState.BEAR_MARKET
                    elif avg_return < -0.01 and avg_volatility >= 0.02:
                        cluster_mapping[cluster_idx] = RegimeState.VOLATILE_BEAR
                    elif avg_volatility > 0.03:
                        cluster_mapping[cluster_idx] = RegimeState.CRISIS
                    elif avg_volatility < 0.015:
                        cluster_mapping[cluster_idx] = RegimeState.SIDEWAYS_LOW_VOL
                    else:
                        cluster_mapping[cluster_idx] = RegimeState.SIDEWAYS_HIGH_VOL
                else:
                    cluster_mapping[cluster_idx] = RegimeState.SIDEWAYS_LOW_VOL
            
            return cluster_mapping
            
        except Exception as e:
            self.logger.error(f"GMM regime mapping failed: {e}")
            return {i: RegimeState.SIDEWAYS_LOW_VOL for i in range(n_clusters)}

    async def _pca_regime_detection(self, features: pd.DataFrame) -> RegimeState:
        """PCA-based regime detection"""
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            
            # Prepare data
            feature_cols = ['returns', 'volatility_20', 'momentum_20']
            available_cols = [col for col in feature_cols if col in features.columns]
            
            if len(available_cols) < 2:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            X = features[available_cols].dropna().values
            
            if len(X) < 10:
                return RegimeState.SIDEWAYS_LOW_VOL
            
            # Apply PCA
            pca = PCA(n_components=min(2, len(available_cols)))
            X_pca = pca.fit_transform(X)
            
            # Cluster in PCA space
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            clusters = kmeans.fit_predict(X_pca)
            
            # Get current regime (last observation)
            current_cluster = clusters[-1]
            
            # Map cluster to regime (simplified)
            current_point = X_pca[-1]
            
            if len(current_point) >= 2:
                pc1, pc2 = current_point[0], current_point[1]
                
                # Simple mapping based on PCA components
                if pc1 > 0 and abs(pc2) < 0.5:
                    return RegimeState.BULL_MARKET
                elif pc1 < 0 and abs(pc2) < 0.5:
                    return RegimeState.BEAR_MARKET
                elif abs(pc1) < 0.5 and pc2 > 0.5:
                    return RegimeState.SIDEWAYS_HIGH_VOL
                elif abs(pc1) < 0.5 and pc2 < -0.5:
                    return RegimeState.SIDEWAYS_LOW_VOL
                else:
                    return RegimeState.VOLATILE_BULL if pc1 > 0 else RegimeState.VOLATILE_BEAR
            
            return RegimeState.SIDEWAYS_LOW_VOL
            
        except Exception as e:
            self.logger.error(f"PCA regime detection failed: {e}")
            return RegimeState.SIDEWAYS_LOW_VOL

    def _calculate_trend_strength_series(self, price_series: pd.Series) -> pd.Series:
        """Calculate trend strength over time"""
        
        try:
            if len(price_series) < 20:
                return pd.Series([0] * len(price_series), index=price_series.index)
            
            trend_strength = []
            
            for i in range(len(price_series)):
                if i < 19:
                    trend_strength.append(0)
                    continue
                
                window_data = price_series.iloc[i-19:i+1]
                
                # Calculate trend using linear regression slope
                x = np.arange(len(window_data))
                y = window_data.values
                
                if len(x) == len(y):
                    slope, _ = np.polyfit(x, y, 1)
                    
                    # Normalize slope to 0-100 scale
                    normalized_slope = min(100, max(0, (slope / np.mean(y)) * 1000 + 50))
                    trend_strength.append(normalized_slope)
                else:
                    trend_strength.append(50)  # Neutral
            
            return pd.Series(trend_strength, index=price_series.index)
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation failed: {e}")
            return pd.Series([50] * len(price_series), index=price_series.index)

    def _detect_volatility_clustering_series(self, returns: pd.Series) -> bool:
        """Detect volatility clustering in return series"""
        
        try:
            if len(returns) < 20:
                return False
            
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=10).std()
            
            if len(rolling_vol.dropna()) < 10:
                return False
            
            # Check for clustering (high volatility followed by high volatility)
            vol_changes = rolling_vol.diff().dropna()
            
            # Count consecutive periods of increasing volatility
            consecutive_increases = 0
            max_consecutive = 0
            
            for change in vol_changes:
                if change > 0:
                    consecutive_increases += 1
                    max_consecutive = max(max_consecutive, consecutive_increases)
                else:
                    consecutive_increases = 0
            
            # Clustering detected if we have 3+ consecutive increases
            return max_consecutive >= 3
            
        except Exception as e:
            self.logger.error(f"Volatility clustering detection failed: {e}")
            return False

    def _extract_cross_asset_features(self, cross_asset_data: Dict) -> pd.DataFrame:
        """Extract features from cross-asset data"""
        
        try:
            features = pd.DataFrame()
            
            # Mock cross-asset features
            if 'correlation_matrix' in cross_asset_data:
                avg_correlation = cross_asset_data['correlation_matrix'].get('average_correlation', 0.5)
                features['cross_asset_correlation'] = [avg_correlation] * 100  # Mock time series
            
            if 'risk_sentiment' in cross_asset_data:
                risk_score = 0.5 if cross_asset_data['risk_sentiment'].get('risk_sentiment') == 'neutral' else 0.8 if cross_asset_data['risk_sentiment'].get('risk_sentiment') == 'risk_off' else 0.2
                features['risk_sentiment_score'] = [risk_score] * 100
            
            if 'flight_to_quality' in cross_asset_data:
                ftq_signal = cross_asset_data['flight_to_quality'].get('flight_to_quality_signal', 0)
                features['flight_to_quality'] = [ftq_signal] * 100
            
            if 'currency_impact' in cross_asset_data:
                currency_strength = 0.5  # Mock neutral
                features['currency_strength'] = [currency_strength] * 100
            
            return features
            
        except Exception as e:
            self.logger.error(f"Cross-asset feature extraction failed: {e}")
            return pd.DataFrame()

    def _extract_macro_features(self, macro_data: Dict) -> pd.DataFrame:
        """Extract features from macroeconomic data"""
        
        try:
            features = pd.DataFrame()
            
            # Mock macro features - in production these would come from economic data
            features['interest_rate_environment'] = [0.07] * 100  # Mock 7% rate environment
            features['inflation_expectation'] = [0.06] * 100      # Mock 6% inflation
            features['economic_growth'] = [0.065] * 100           # Mock 6.5% growth
            features['policy_uncertainty'] = [0.3] * 100          # Mock moderate uncertainty
            
            # GDP growth trend
            features['gdp_trend'] = [1 if i > 50 else -1 for i in range(100)]  # Mock trend
            
            # Employment indicators
            features['employment_strength'] = [0.7] * 100         # Mock strong employment
            
            # Global factors
            features['global_risk_appetite'] = [0.6] * 100        # Mock moderate risk appetite
            features['commodity_price_pressure'] = [0.4] * 100    # Mock moderate pressure
            
            return features
            
        except Exception as e:
            self.logger.error(f"Macro feature extraction failed: {e}")
            return pd.DataFrame()