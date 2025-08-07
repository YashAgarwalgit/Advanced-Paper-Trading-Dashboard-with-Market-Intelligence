"""
Professional order management system with institutional-grade features
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from config import Config
from utils.helpers import get_timestamp_iso
from utils.decorators import async_retry_on_failure

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    BRACKET = "BRACKET"
    OCO = "OCO"  # One-Cancels-Other
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

@dataclass
class OrderRequest:
    """Comprehensive order request structure"""
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Advanced order parameters
    iceberg_quantity: Optional[float] = None
    twap_duration: Optional[int] = None  # Minutes
    bracket_profit_price: Optional[float] = None
    bracket_stop_price: Optional[float] = None
    
    # Risk parameters
    max_slippage_pct: float = 0.5
    max_position_size: Optional[float] = None
    
    def __post_init__(self):
        self.order_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()

@dataclass
class Order:
    """Order state management with comprehensive tracking"""
    request: OrderRequest
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: float = 0.0
    total_fill_value: float = 0.0
    commission: float = 0.0
    
    # Execution tracking
    fills: List[Dict] = field(default_factory=list)
    status_history: List[Dict] = field(default_factory=list)
    
    # Child orders (for bracket, OCO, etc.)
    child_orders: List['Order'] = field(default_factory=list)
    parent_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.request.quantity
        
        self.status_history.append({
            "status": self.status,
            "timestamp": get_timestamp_iso(),
            "message": "Order created"
        })
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
    @property
    def fill_percentage(self) -> float:
        return (self.filled_quantity / self.request.quantity) * 100
    
    def add_fill(self, quantity: float, price: float, commission: float = 0.0):
        """Add execution fill to order"""
        fill = {
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "timestamp": get_timestamp_iso(),
            "fill_id": str(uuid.uuid4())
        }
        
        self.fills.append(fill)
        self.filled_quantity += quantity
        self.remaining_quantity -= quantity
        self.total_fill_value += quantity * price
        self.commission += commission
        
        # Update average fill price
        if self.filled_quantity > 0:
            self.average_fill_price = self.total_fill_value / self.filled_quantity
        
        # Update status
        if self.remaining_quantity <= 0.001:
            self.update_status(OrderStatus.FILLED, f"Order fully filled at avg price {self.average_fill_price:.2f}")
        elif self.filled_quantity > 0:
            self.update_status(OrderStatus.PARTIAL_FILLED, f"Partial fill: {quantity} @ {price}")
    
    def update_status(self, new_status: OrderStatus, message: str = ""):
        """Update order status with history tracking"""
        self.status = new_status
        self.status_history.append({
            "status": new_status,
            "timestamp": get_timestamp_iso(),
            "message": message
        })

class ProfessionalOrderManager:
    """
    Institutional-grade order management system
    Features: Multiple order types, execution algorithms, risk management
    """
    
    def __init__(self, market_data_manager):
        self.market_data_manager = market_data_manager
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Risk limits from config
        self.max_order_value = Config.MAX_ORDER_VALUE
        self.max_daily_turnover = Config.MAX_DAILY_TURNOVER
        
        # Execution algorithms
        self.execution_algorithms = {
            OrderType.TWAP: self._execute_twap,
            OrderType.VWAP: self._execute_vwap,
            OrderType.ICEBERG: self._execute_iceberg,
        }
        
        # Start order monitoring
        self._start_order_monitoring()
    
    async def place_order(self, order_request: OrderRequest, portfolio: Dict) -> Tuple[bool, str, Optional[Order]]:
        """
        Place order with comprehensive validation and risk checks
        
        Returns:
            (success, message, order_object)
        """
        
        try:
            # Validate order request
            validation_result = self._validate_order_request(order_request, portfolio)
            if not validation_result[0]:
                return False, validation_result[1], None
            
            # Create order object
            order = Order(request=order_request)
            
            # Route order based on type
            routing_success = await self._route_order(order, portfolio)
            if not routing_success[0]:
                order.update_status(OrderStatus.REJECTED, routing_success[1])
                return False, routing_success[1], order
            
            # Add to active orders
            self.active_orders[order.request.order_id] = order
            
            self.logger.info(f"Order placed successfully: {order.request.order_id}")
            return True, f"Order {order.request.order_id} placed successfully", order
            
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return False, f"Order placement error: {e}", None
    
    def _validate_order_request(self, request: OrderRequest, portfolio: Dict) -> Tuple[bool, str]:
        """Comprehensive order validation"""
        
        # Basic parameter validation
        if request.quantity <= 0:
            return False, "Quantity must be positive"
        
        if request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and request.price is None:
            return False, f"{request.order_type.value} orders require a price"
        
        if request.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT] and request.stop_price is None:
            return False, f"{request.order_type.value} orders require a stop price"
        
        # Order size validation
        if request.price:
            order_value = request.quantity * request.price
            if order_value > self.max_order_value:
                return False, f"Order value {order_value:,.0f} exceeds maximum {self.max_order_value:,.0f}"
        
        # Portfolio-specific validations
        if request.side == OrderSide.SELL:
            current_position = portfolio.get('positions', {}).get(request.ticker, {})
            available_quantity = current_position.get('quantity', 0)
            
            if request.quantity > available_quantity:
                return False, f"Insufficient position: have {available_quantity}, trying to sell {request.quantity}"
        
        return True, "Validation passed"
    
    async def _route_order(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Route order to appropriate execution method"""
        
        try:
            request = order.request
            
            # Market orders - immediate execution
            if request.order_type == OrderType.MARKET:
                return await self._execute_market_order(order, portfolio)
            
            # Algorithmic orders
            elif request.order_type in self.execution_algorithms:
                algorithm = self.execution_algorithms[request.order_type]
                return await algorithm(order, portfolio)
            
            # Standard orders - add to order book
            elif request.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                return await self._add_to_order_book(order, portfolio)
            
            # Complex orders
            elif request.order_type == OrderType.BRACKET:
                return await self._create_bracket_order(order, portfolio)
            
            else:
                return False, f"Unsupported order type: {request.order_type.value}"
                
        except Exception as e:
            self.logger.error(f"Order routing failed: {e}")
            return False, f"Routing error: {e}"
    
    async def _execute_market_order(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Execute market order immediately"""
        
        try:
            from .trading_engine import AdvancedTradingEngine
            
            request = order.request
            
            # Get current price
            success, prices = await self.market_data_manager.get_live_prices_async([request.ticker])
            if not success or request.ticker not in prices:
                return False, f"Unable to get market price for {request.ticker}"
            
            market_price = prices[request.ticker]
            
            # Execute through trading engine
            trading_engine = AdvancedTradingEngine()
            result = await trading_engine.execute_trade_async(
                portfolio, request.ticker, request.side.value, 
                request.quantity, market_price, request.max_slippage_pct
            )
            
            if result["success"]:
                # Record fill
                order.add_fill(request.quantity, result["execution_price"], result["commission"])
                return True, result["message"]
            else:
                return False, result["message"]
                
        except Exception as e:
            self.logger.error(f"Market order execution failed: {e}")
            return False, f"Execution error: {e}"
    
    async def _execute_twap(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Execute TWAP (Time-Weighted Average Price) algorithm"""
        
        try:
            request = order.request
            duration_minutes = request.twap_duration or 30
            
            # Calculate slice parameters
            num_slices = min(10, duration_minutes // 3)
            slice_size = request.quantity / num_slices
            interval_seconds = (duration_minutes * 60) / num_slices
            
            order.update_status(OrderStatus.PENDING, f"TWAP execution started: {num_slices} slices over {duration_minutes} minutes")
            
            # Schedule execution slices
            asyncio.create_task(self._execute_twap_slices(
                order, portfolio, slice_size, interval_seconds, num_slices
            ))
            
            return True, "TWAP algorithm initiated"
            
        except Exception as e:
            self.logger.error(f"TWAP setup failed: {e}")
            return False, f"TWAP error: {e}"
    
    async def _execute_twap_slices(self, order: Order, portfolio: Dict, 
                                 slice_size: float, interval_seconds: float, num_slices: int):
        """Execute TWAP slices over time"""
        
        from .trading_engine import AdvancedTradingEngine
        trading_engine = AdvancedTradingEngine()
        
        for slice_num in range(num_slices):
            try:
                if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    break
                
                # Wait for next slice (except first one)
                if slice_num > 0:
                    await asyncio.sleep(interval_seconds)
                
                # Get current price and execute slice
                success, prices = await self.market_data_manager.get_live_prices_async([order.request.ticker])
                if not success or order.request.ticker not in prices:
                    continue
                
                market_price = prices[order.request.ticker]
                
                # Execute slice
                result = await trading_engine.execute_trade_async(
                    portfolio, order.request.ticker, order.request.side.value,
                    slice_size, market_price, order.request.max_slippage_pct
                )
                
                if result["success"]:
                    order.add_fill(slice_size, result["execution_price"], result["commission"])
                    self.logger.info(f"TWAP slice {slice_num + 1}/{num_slices} executed at {result['execution_price']:.2f}")
                
            except Exception as e:
                self.logger.error(f"TWAP slice execution failed: {e}")
                continue
    
    async def _execute_vwap(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Execute VWAP algorithm (simplified implementation)"""
        # For now, fall back to TWAP with volume weighting consideration
        return await self._execute_twap(order, portfolio)
    
    async def _execute_iceberg(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Execute iceberg order"""
        request = order.request
        iceberg_size = request.iceberg_quantity or (request.quantity * 0.1)  # 10% default
        
        # Split into smaller visible orders
        remaining = request.quantity
        while remaining > 0:
            current_size = min(iceberg_size, remaining)
            
            # Create child order for this slice
            child_request = OrderRequest(
                ticker=request.ticker,
                side=request.side,
                quantity=current_size,
                order_type=OrderType.LIMIT,
                price=request.price
            )
            
            child_order = Order(request=child_request)
            child_order.parent_order_id = order.request.order_id
            order.child_orders.append(child_order)
            
            remaining -= current_size
        
        return True, f"Iceberg order created with {len(order.child_orders)} slices"
    
    async def _add_to_order_book(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Add order to order book for later execution"""
        # Implementation would depend on exchange connectivity
        # For now, mark as pending
        order.update_status(OrderStatus.PENDING, "Order added to book")
        return True, "Order added to order book"
    
    async def cleanup_monitoring_task(self):
        """Cleanup monitoring task to prevent pending task warnings"""
        
        try:
            if hasattr(self, '_monitoring_task') and self._monitoring_task:
                if not self._monitoring_task.done():
                    self._monitoring_task.cancel()
                    try:
                        await self._monitoring_task
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling
                self._monitoring_task = None
                self.logger.info("Order monitoring task cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Error cleaning up monitoring task: {e}")
    
    async def _create_bracket_order(self, order: Order, portfolio: Dict) -> Tuple[bool, str]:
        """Create bracket order with profit target and stop loss"""
        request = order.request
        
        if not request.bracket_profit_price or not request.bracket_stop_price:
            return False, "Bracket orders require profit and stop prices"
        
        # Create main order
        main_request = OrderRequest(
            ticker=request.ticker,
            side=request.side,
            quantity=request.quantity,
            order_type=OrderType.LIMIT,
            price=request.price
        )
        
        # Create profit target
        profit_side = OrderSide.SELL if request.side == OrderSide.BUY else OrderSide.BUY
        profit_request = OrderRequest(
            ticker=request.ticker,
            side=profit_side,
            quantity=request.quantity,
            order_type=OrderType.LIMIT,
            price=request.bracket_profit_price
        )
        
        # Create stop loss
        stop_request = OrderRequest(
            ticker=request.ticker,
            side=profit_side,
            quantity=request.quantity,
            order_type=OrderType.STOP_LOSS,
            stop_price=request.bracket_stop_price
        )
        
        # Link orders
        main_order = Order(request=main_request)
        profit_order = Order(request=profit_request)
        stop_order = Order(request=stop_request)
        
        main_order.child_orders = [profit_order, stop_order]
        profit_order.parent_order_id = main_order.request.order_id
        stop_order.parent_order_id = main_order.request.order_id
        
        return True, "Bracket order created successfully"
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel active order"""
        
        if order_id not in self.active_orders:
            return False, "Order not found"
        
        order = self.active_orders[order_id]
        
        if order.is_complete:
            return False, f"Order already {order.status.value}"
        
        order.update_status(OrderStatus.CANCELLED, "Order cancelled by user")
        
        # Move to history
        self.order_history.append(order)
        del self.active_orders[order_id]
        
        return True, f"Order {order_id} cancelled successfully"
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get comprehensive order status"""
        
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            return {
                "order_id": order_id,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity,
                "fill_percentage": order.fill_percentage,
                "average_fill_price": order.average_fill_price,
                "total_commission": order.commission,
                "fills": order.fills,
                "status_history": order.status_history
            }
        
        return None
    
    def get_order_book(self) -> Dict[str, List[Dict]]:
        """Get current order book"""
        
        buy_orders = []
        sell_orders = []
        
        for order in self.active_orders.values():
            order_info = {
                "order_id": order.request.order_id,
                "ticker": order.request.ticker,
                "quantity": order.remaining_quantity,
                "price": order.request.price,
                "order_type": order.request.order_type.value,
                "timestamp": order.request.timestamp.isoformat(),
                "status": order.status.value
            }
            
            if order.request.side == OrderSide.BUY:
                buy_orders.append(order_info)
            else:
                sell_orders.append(order_info)
        
        return {
            "buy_orders": sorted(buy_orders, key=lambda x: x['price'] or 0, reverse=True),
            "sell_orders": sorted(sell_orders, key=lambda x: x['price'] or float('inf'))
        }
    
    def _start_order_monitoring(self):
        """Start background order monitoring"""
        asyncio.create_task(self._monitor_orders())
    
    async def _monitor_orders(self):
        """Monitor active orders for fills and expiry"""
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = datetime.utcnow()
                expired_orders = []
                
                for order_id, order in self.active_orders.items():
                    # Check for expiry
                    if order.request.time_in_force == TimeInForce.DAY:
                        if (current_time - order.request.timestamp) > timedelta(hours=8):
                            expired_orders.append(order_id)
                
                # Handle expired orders
                for order_id in expired_orders:
                    order = self.active_orders[order_id]
                    order.update_status(OrderStatus.EXPIRED, "Order expired")
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                
            except Exception as e:
                self.logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(1)
