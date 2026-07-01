"""
rabbitmq_client.py - مدیریت اتصال به RabbitMQ
"""

import aio_pika
from typing import Optional, Callable, Any
import json
import logging
from .config import settings

logger = logging.getLogger(__name__)

_connection: Optional[aio_pika.Connection] = None
_channel: Optional[aio_pika.Channel] = None


async def init_rabbitmq():
    """مقداردهی اولیه اتصال RabbitMQ"""
    global _connection, _channel
    
    try:
        _connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
        _channel = await _connection.channel()
        logger.info("✅ اتصال به RabbitMQ برقرار شد")
        return _connection
    except Exception as e:
        logger.error(f"❌ خطا در اتصال به RabbitMQ: {e}")
        raise


async def close_rabbitmq():
    """بستن اتصال RabbitMQ"""
    global _connection, _channel
    if _channel:
        await _channel.close()
    if _connection:
        await _connection.close()
        logger.info("✅ اتصال RabbitMQ بسته شد")


def get_channel() -> aio_pika.Channel:
    """دریافت کانال RabbitMQ"""
    if _channel is None:
        raise Exception("RabbitMQ not initialized. Call init_rabbitmq() first.")
    return _channel


class RabbitMQClient:
    """کلاس wrapper برای عملیات RabbitMQ"""
    
    @staticmethod
    async def publish(queue: str, message: Any, reply_to: Optional[str] = None):
        """ارسال پیام به صف"""
        channel = get_channel()
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                reply_to=reply_to
            ),
            routing_key=queue
        )
        logger.debug(f"پیام به صف {queue} ارسال شد")
    
    @staticmethod
    async def consume(queue: str, callback: Callable):
        """مصرف پیام از صف"""
        channel = get_channel()
        queue_obj = await channel.declare_queue(queue, durable=True)
        await queue_obj.consume(callback)
        logger.info(f"شروع مصرف از صف {queue}")
    
    @staticmethod
    async def declare_queue(queue: str, durable: bool = True):
        """ایجاد صف"""
        channel = get_channel()
        return await channel.declare_queue(queue, durable=durable)


rabbitmq_client = RabbitMQClient()