"""Chat channels module with plugin architecture."""

from clicomp.channels.base import BaseChannel
from clicomp.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
