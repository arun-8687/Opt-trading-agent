"""Order lifecycle management — placement, modification, cancellation, and fill tracking."""

import time as time_module
from datetime import datetime
from typing import Optional

from src.broker.base import BaseBroker
from src.broker.models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
)
from src.strategies.base import TradeSetup
from src.utils.logger import get_logger

logger = get_logger("order_manager")


class OrderManager:
    """Manages order placement, tracking, and modification."""

    def __init__(
        self,
        broker: BaseBroker,
        mode: str = "paper",
        max_retry_seconds: int = 30,
    ):
        self.broker = broker
        self.mode = mode  # "paper" or "live"
        self.max_retry_seconds = max_retry_seconds
        self._orders: dict[str, Order] = {}
        self._paper_order_counter = 0

    def place_entry_order(self, setup: TradeSetup) -> Optional[Order]:
        """Place an entry order for a trade setup.

        Uses LIMIT order at current market price + 1 tick buffer.
        Falls back to MARKET if not filled within timeout.
        """
        contract = setup.contract

        order = Order(
            symbol=setup.symbol,
            trading_symbol=contract.trading_symbol,
            token=contract.token,
            exchange=contract.exchange,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            product=ProductType.INTRADAY,
            quantity=setup.quantity,
            price=self._get_entry_price(setup.entry_price),
            tag=setup.strategy_name,
        )

        if self.mode == "paper":
            return self._paper_place(order, setup.entry_price)

        # Live order
        order_id = self.broker.place_order(order)
        if not order_id:
            logger.error(f"ORDER FAILED: {contract.display_name}")
            return None

        order.order_id = order_id
        self._orders[order_id] = order

        # Wait for fill, modify to market if timeout
        filled = self._wait_for_fill(order)
        if not filled:
            logger.warning(f"Order not filled, converting to MARKET: {order_id}")
            self.broker.modify_order(order_id, order_type="MARKET")
            self._wait_for_fill(order, timeout=10)

        return order

    def place_exit_order(
        self,
        trading_symbol: str,
        token: str,
        exchange: str,
        quantity: int,
        price: float = 0.0,
        reason: str = "",
    ) -> Optional[Order]:
        """Place an exit (sell) order."""
        order_type = OrderType.LIMIT if price > 0 else OrderType.MARKET

        order = Order(
            symbol=trading_symbol,
            trading_symbol=trading_symbol,
            token=token,
            exchange=exchange,
            side=OrderSide.SELL,
            order_type=order_type,
            product=ProductType.INTRADAY,
            quantity=quantity,
            price=price,
            tag=f"exit_{reason}",
        )

        if self.mode == "paper":
            return self._paper_place(order, price if price > 0 else 0)

        order_id = self.broker.place_order(order)
        if not order_id:
            # Critical: exit failed, try market order
            logger.error(f"EXIT ORDER FAILED for {trading_symbol}, retrying as MARKET")
            order.order_type = OrderType.MARKET
            order.price = 0
            order_id = self.broker.place_order(order)

        if order_id:
            order.order_id = order_id
            self._orders[order_id] = order
            logger.info(
                f"EXIT ORDER: SELL {trading_symbol} qty={quantity} "
                f"reason={reason} id={order_id}"
            )

        return order if order_id else None

    def place_sl_order(
        self,
        trading_symbol: str,
        token: str,
        exchange: str,
        quantity: int,
        trigger_price: float,
        limit_price: float,
    ) -> Optional[Order]:
        """Place a stop-loss order."""
        order = Order(
            symbol=trading_symbol,
            trading_symbol=trading_symbol,
            token=token,
            exchange=exchange,
            side=OrderSide.SELL,
            order_type=OrderType.STOPLOSS_LIMIT,
            product=ProductType.INTRADAY,
            quantity=quantity,
            price=limit_price,
            trigger_price=trigger_price,
            tag="sl_order",
        )

        if self.mode == "paper":
            return self._paper_place(order, trigger_price)

        order_id = self.broker.place_order(order)
        if order_id:
            order.order_id = order_id
            self._orders[order_id] = order
            logger.info(
                f"SL ORDER: {trading_symbol} trigger={trigger_price} "
                f"limit={limit_price} id={order_id}"
            )
        return order if order_id else None

    def modify_sl_order(
        self,
        order_id: str,
        new_trigger_price: float,
        new_limit_price: float,
    ) -> bool:
        """Modify an existing SL order with new trigger price."""
        if self.mode == "paper":
            if order_id in self._orders:
                self._orders[order_id].trigger_price = new_trigger_price
                self._orders[order_id].price = new_limit_price
                return True
            return False

        return self.broker.modify_order(
            order_id,
            trigger_price=new_trigger_price,
            price=new_limit_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.mode == "paper":
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
                return True
            return False

        return self.broker.cancel_order(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def _get_entry_price(self, ltp: float) -> float:
        """Calculate entry limit price (LTP + small buffer for fill)."""
        # Add 0.5% buffer above LTP for buy orders
        buffer = max(0.05, ltp * 0.005)
        return round(ltp + buffer, 2)

    def _wait_for_fill(self, order: Order, timeout: int = 0) -> bool:
        """Wait for order fill up to timeout seconds."""
        timeout = timeout or self.max_retry_seconds
        start = time_module.time()

        while time_module.time() - start < timeout:
            order_book = self.broker.get_order_book()
            for ob_order in order_book:
                if str(ob_order.get("orderid")) == order.order_id:
                    status = ob_order.get("status", "").lower()
                    if status == "complete":
                        order.status = OrderStatus.COMPLETE
                        order.filled_price = float(ob_order.get("averageprice", 0))
                        order.filled_quantity = int(ob_order.get("filledshares", 0))
                        logger.info(
                            f"ORDER FILLED: {order.trading_symbol} "
                            f"@ {order.filled_price} qty={order.filled_quantity}"
                        )
                        return True
                    elif status == "rejected":
                        order.status = OrderStatus.REJECTED
                        logger.error(
                            f"ORDER REJECTED: {order.trading_symbol} "
                            f"reason={ob_order.get('text', 'unknown')}"
                        )
                        return False
            time_module.sleep(1)

        return False

    def _paper_place(self, order: Order, fill_price: float) -> Order:
        """Simulate order placement in paper trading mode."""
        self._paper_order_counter += 1
        order.order_id = f"PAPER_{self._paper_order_counter}"
        order.status = OrderStatus.COMPLETE
        order.filled_price = fill_price if fill_price > 0 else order.price
        order.filled_quantity = order.quantity
        order.timestamp = datetime.now()

        self._orders[order.order_id] = order

        logger.info(
            f"PAPER ORDER: {order.side.value} {order.trading_symbol} "
            f"@ {order.filled_price} qty={order.quantity} id={order.order_id}"
        )

        return order
