import asyncio
import logging
from typing import Any, Dict, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message, Update, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.bot import DefaultBotProperties
from pydantic import ValidationError

import app.config as config

logger = logging.getLogger("telegram_bot")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

TELEGRAM_ENABLED = bool(config.TELEGRAM_TOKEN)

bot: Optional[Bot] = None
dp: Optional[Dispatcher] = None
_polling_task: Optional[asyncio.Task] = None

if TELEGRAM_ENABLED:
    bot = Bot(
        token=config.TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    @dp.message()
    async def handle_any_message(message: Message) -> None:
        """Log chat id for convenience when the flag is enabled."""
        if config.TELEGRAM_PRINT_CHAT_ID:
            logger.info(
                "chat_id=%s username=%s text=%s",
                message.chat.id,
                message.from_user.username if message.from_user else None,
                message.text,
            )

    @dp.callback_query(F.data.startswith("bet_size:"))
    async def handle_bet_size_callback(callback: CallbackQuery) -> None:
        """Handle inline button actions for bet size confirmation."""
        data = (callback.data or "").split(":")
        if len(data) != 3:
            await callback.answer("Некорректное действие", show_alert=True)
            return

        _, request_id, action = data
        from_user = callback.from_user
        actor = None
        if from_user:
            if from_user.username:
                actor = f"@{from_user.username}"
            else:
                full_name = " ".join(filter(None, [from_user.first_name, from_user.last_name])).strip()
                actor = full_name or str(from_user.id)

        from predictor import poly_service  # Local import to avoid circular dependency

        if action == "approve":
            result = await poly_service.approve_bet_size_request(request_id, actor)
        elif action == "reject":
            result = await poly_service.reject_bet_size_request(request_id, actor, reason="telegram_reject")
        else:
            await callback.answer("Неизвестное действие", show_alert=True)
            return

        status = (result or {}).get("status")
        if status == "approved":
            await callback.answer("Размер ставки обновлён")
        elif status == "rejected":
            await callback.answer("Запрос отменён")
        elif status == "expired":
            await callback.answer("Запрос уже истёк", show_alert=True)
        else:
            await callback.answer("Запрос не найден", show_alert=True)

        await _finalize_callback_message(callback, result, action)
else:
    logger.warning("Telegram bot disabled (no TELEGRAM_TOKEN)")


def _bot_ready() -> bool:
    return TELEGRAM_ENABLED and bot is not None and dp is not None


def start_polling() -> None:
    global _polling_task
    if not _bot_ready():
        logger.info("Telegram polling not started: disabled or missing token")
        return
    if _polling_task and not _polling_task.done():
        return

    assert bot and dp

    async def _runner():
        global _polling_task
        logger.info("Starting Telegram polling loop")
        try:
            await dp.start_polling(bot)
        except asyncio.CancelledError:
            logger.info("Telegram polling cancelled")
            raise
        except Exception as exc:
            logger.error("Telegram polling stopped with error: %s", exc)
        finally:
            _polling_task = None
            logger.info("Telegram polling loop exited")

    loop = asyncio.get_event_loop()
    _polling_task = loop.create_task(_runner(), name="telegram-polling")


async def stop_polling() -> None:
    global _polling_task
    if not _polling_task:
        return
    _polling_task.cancel()
    try:
        await _polling_task
    except asyncio.CancelledError:
        pass
    _polling_task = None


async def process_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Feed raw Telegram update into aiogram dispatcher."""
    if not _bot_ready():
        return {"status": "telegram_disabled"}
    assert bot and dp
    try:
        update = Update.model_validate(payload)
    except ValidationError as exc:
        logger.error("Invalid telegram update: %s", exc)
        return {"status": "invalid_update"}

    await dp.feed_update(bot, update)
    return {"status": "ok"}


async def send_message(
    chat_id: str | int,
    text: str,
    reply_markup: Any | None = None,
) -> bool:
    if not _bot_ready():
        logger.warning("send_message skipped: telegram disabled")
        return False
    assert bot
    try:
        await bot.send_message(chat_id, text, reply_markup=reply_markup)
        return True
    except TelegramAPIError as exc:
        logger.error("Failed to send telegram message: %s", exc)
        return False


async def notify_info_chats(text: str) -> None:
    if not config.TELEGRAM_INFO_CHAT_IDS:
        return
    for chat_id in config.TELEGRAM_INFO_CHAT_IDS:
        await send_message(chat_id, text)


async def notify_admin(text: str, reply_markup: Any | None = None) -> bool:
    if not config.TELEGRAM_ADMIN_CHAT_ID:
        logger.warning("No TELEGRAM_ADMIN_CHAT_ID configured")
        return False
    return await send_message(config.TELEGRAM_ADMIN_CHAT_ID, text, reply_markup)


def build_bet_size_keyboard(request_id: str) -> Any:
    builder = InlineKeyboardBuilder()
    builder.button(text="Подтвердить", callback_data=f"bet_size:{request_id}:approve")
    builder.button(text="Отмена", callback_data=f"bet_size:{request_id}:reject")
    builder.adjust(2)
    return builder.as_markup()


def _format_bet_size_resolution_text(result: Optional[Dict[str, Any]] | None, action: str | None) -> str:
    data = result or {}
    status = (data.get("status") or "").lower()

    selection_text = {
        "approve": "Вы подтвердили изменение ставки.",
        "reject": "Вы отменили изменение ставки.",
    }.get(action, "Действие обработано.")

    def _fmt_amount(value: Any) -> str:
        try:
            return f"{float(value):.2f}$"
        except (TypeError, ValueError):
            return str(value) if value is not None else "н/д"

    lines: list[str] = [selection_text]

    if status == "approved":
        new_size = _fmt_amount(data.get("requested_bet_size"))
        prev_size = data.get("previous_bet_size")
        delta_line = f"Новый размер: {new_size}"
        if prev_size is not None:
            delta_line += f" (было {_fmt_amount(prev_size)})"
        lines.append("✅ Размер ставки обновлён.")
        lines.append(delta_line)
    elif status == "rejected":
        lines.append("⛔️ Запрос на изменение ставки отклонён.")
    elif status == "expired":
        lines.append("⏱ Время подтверждения истекло.")
    elif status == "missing":
        lines.append("⚠️ Запрос не найден.")
    elif status:
        lines.append(f"ℹ️ Статус запроса: {status}.")

    balance_value = data.get("balance_display")
    if balance_value is not None:
        try:
            balance_line = f"Текущий баланс: {float(balance_value):.2f}$"
        except (TypeError, ValueError):
            balance_line = f"Текущий баланс: {balance_value}"
        lines.append(balance_line)

    return "\n".join(filter(None, lines))


async def _finalize_callback_message(
    callback: CallbackQuery,
    result: Optional[Dict[str, Any]] | None,
    action: str | None,
) -> None:
    message = callback.message
    if not message:
        return

    text = _format_bet_size_resolution_text(result, action)

    try:
        await message.edit_text(text, reply_markup=None)
    except TelegramAPIError as exc:
        logger.warning("Failed to edit telegram confirmation message: %s", exc)
        try:
            await message.edit_reply_markup(reply_markup=None)
        except TelegramAPIError:
            pass
