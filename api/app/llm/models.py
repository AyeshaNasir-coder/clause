from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    return init_chat_model(
        model=settings.chat_model,
        api_key=settings.chat_api_key,
        model_provider=settings.chat_provider,
        # temperature=settings.chat_temperature,
    )


def get_fallback_model() -> None:
    pass