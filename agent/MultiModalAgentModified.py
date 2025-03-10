from __future__ import annotations

import logging

from livekit.agents.multimodal import MultimodalAgent

try:
    from tools import AssistantFnc
    assistantFnc = AssistantFnc()
except ImportError:
    AssistantFnc = None
    assistantFnc = None


from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


from typing import Literal



class MultimodalAgentModified(MultimodalAgent):
    """
    Overwrite certain method to make it work with None Vad mode
    """
    def generate_reply(
        self,
        on_duplicate: Literal[
            "cancel_existing", "cancel_new", "keep_both"
        ] = "cancel_existing",
        enable_vad: bool = True
    ) -> None:
        """Generate a reply from the agent"""

        # remove duplicated reply when vad disabled
        if not enable_vad:
            self._session.commit_audio_buffer()
        self._session.create_response(on_duplicate=on_duplicate)