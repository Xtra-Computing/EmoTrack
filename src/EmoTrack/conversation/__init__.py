"""Online conversation runtime."""

from EmoTrack.conversation.config import ConversationConfig, load_conversation_config


def run_online_conversation_sessions(*args, **kwargs):  # noqa: ANN002, ANN003
    from EmoTrack.conversation.api import run_online_conversation_sessions as _run_online_conversation_sessions

    return _run_online_conversation_sessions(*args, **kwargs)


def run_conversation_sessions(*args, **kwargs):  # noqa: ANN002, ANN003
    from EmoTrack.conversation.api import run_conversation_sessions as _run_conversation_sessions

    return _run_conversation_sessions(*args, **kwargs)


__all__ = [
    "ConversationConfig",
    "load_conversation_config",
    "run_conversation_sessions",
    "run_online_conversation_sessions",
]
