"""Message bus module for decoupled channel-agent communication."""

from clicomp.bus.events import InboundMessage, OutboundMessage
from clicomp.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
