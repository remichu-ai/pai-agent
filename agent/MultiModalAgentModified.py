from __future__ import annotations

import logging

from livekit.agents.multimodal import MultimodalAgent, AgentTranscriptionOptions

try:
    from tools import AssistantFnc
    assistantFnc = AssistantFnc()
except ImportError:
    AssistantFnc = None
    assistantFnc = None





from livekit import api
from data_model import GallamaSessionConfig, TurnDetectionConfig
import functools

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


from livekit.agents import llm, transcription, vad
from typing import Literal, Protocol, AsyncIterable

import websockets
import struct
from livekit.agents.utils.images import encode, EncodeOptions, ResizeOptions
import os
from PIL import Image
import io
import time
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from livekit.agents import stt




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

    # def _setup_track(self, track):
    #     self._subscribed_track = track
    #     self._stt_forwarder = transcription.STTSegmentsForwarder(
    #         room=self._room,
    #         participant=self._linked_participant,
    #         track=self._subscribed_track,
    #     )
    #
    #     if self._read_micro_atask is not None:
    #         self._read_micro_atask.cancel()
    #
    #     self._read_micro_atask = asyncio.create_task(
    #         self._micro_task(self._subscribed_track)
    #     )
    #     logger.info(f"Successfully subscribed to track {track.sid} for participant {self._linked_participant.identity}")
    #
    # def _subscribe_to_microphone(self, *args, **kwargs) -> None:
    #     """Subscribe to the participant microphone if found"""
    #
    #     if self._linked_participant is None:
    #         return
    #
    #     # Check if the arguments include a track publication (happens when track_published event fires)
    #     if len(args) > 0 and isinstance(args[0], rtc.RemoteTrackPublication):
    #         publication = args[0]
    #         participant = publication.participant
    #
    #         # Only process if this is for our linked participant
    #         if participant.identity != self._linked_participant.identity:
    #             return
    #
    #         # Only handle microphone tracks
    #         if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
    #             return
    #
    #         # Subscribe to the track
    #         if not publication.subscribed:
    #             publication.set_subscribed(True)
    #
    #         # If track is available, set it up
    #         if publication.track is not None and publication.track != self._subscribed_track:
    #             self._setup_track(publication.track)
    #         return
    #
    #     # Regular check for existing tracks (happens during initialization)
    #     for publication in self._linked_participant.track_publications.values():
    #         if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
    #             continue
    #
    #         if not publication.subscribed:
    #             publication.set_subscribed(True)
    #
    #         if publication.track is not None and publication.track != self._subscribed_track:
    #             self._setup_track(publication.track)
    #             break
    #
    # def _on_track_subscribed(self, track, publication, participant):
    #     """Handle track subscription events for microphone tracks"""
    #     logger.info(f"Track subscribed: {track.sid}, source: {publication.source}")
    #
    #     # Only process microphone tracks for the linked participant
    #     if (self._linked_participant and
    #             publication.source == rtc.TrackSource.SOURCE_MICROPHONE and
    #             participant.identity == self._linked_participant.identity and
    #             track != self._subscribed_track):
    #
    #         logger.info(f"Setting up microphone track for participant {participant.identity}")
    #         self._subscribed_track = track
    #
    #         # Create STT forwarder for the track
    #         self._stt_forwarder = transcription.STTSegmentsForwarder(
    #             room=self._room,
    #             participant=self._linked_participant,
    #             track=self._subscribed_track,
    #         )
    #
    #         # Cancel existing microphone task if running
    #         if self._read_micro_atask is not None:
    #             self._read_micro_atask.cancel()
    #
    #         # Start a new microphone task
    #         self._read_micro_atask = asyncio.create_task(
    #             self._micro_task(self._subscribed_track)
    #         )
    #
    # def start(
    #         self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    # ) -> None:
    #     if self._started:
    #         raise RuntimeError("voice assistant already started")
    #
    #     room.on("participant_connected", self._on_participant_connected)
    #     room.on("track_published", self._subscribe_to_microphone)
    #     room.on("track_subscribed", self._on_track_subscribed)  # Add this line
    #
    #     self._room, self._participant = room, participant
    #
    #     if participant is not None:
    #         if isinstance(participant, rtc.RemoteParticipant):
    #             self._link_participant(participant.identity)
    #         else:
    #             self._link_participant(participant)
    #     else:
    #         # no participant provided, try to find the first participant in the room
    #         for participant in self._room.remote_participants.values():
    #             self._link_participant(participant.identity)
    #             break
    #
    #     self._session = self._model.session(
    #         chat_ctx=self._chat_ctx, fnc_ctx=self._fnc_ctx
    #     )
    #
    #     # Create a task to wait for initialization and start the main task
    #     async def _init_and_start():
    #         try:
    #             await self._session._init_sync_task
    #             logger.info("Session initialized with chat context")
    #             self._main_atask = asyncio.create_task(self._main_task())
    #         except Exception as e:
    #             logger.exception("Failed to initialize session")
    #             raise e
    #
    #     # Schedule the initialization and start task
    #     asyncio.create_task(_init_and_start())
    #
    #     @self._session.on("response_content_added")
    #     def _on_content_added(message: _ContentProto):
    #         tr_fwd = transcription.TTSSegmentsForwarder(
    #             room=self._room,
    #             participant=self._room.local_participant,
    #             speed=self._opts.transcription.agent_transcription_speed,
    #             sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
    #             word_tokenizer=self._opts.transcription.word_tokenizer,
    #             hyphenate_word=self._opts.transcription.hyphenate_word,
    #         )
    #
    #         self._playing_handle = self._agent_playout.play(
    #             item_id=message.item_id,
    #             content_index=message.content_index,
    #             transcription_fwd=tr_fwd,
    #             text_stream=message.text_stream,
    #             audio_stream=message.audio_stream,
    #         )
    #
    #     @self._session.on("response_content_done")
    #     def _response_content_done(message: _ContentProto):
    #         if message.content_type == "text":
    #             if self._text_response_retries >= self._max_text_response_retries:
    #                 raise RuntimeError(
    #                     f"The OpenAI Realtime API returned a text response "
    #                     f"after {self._max_text_response_retries} retries. "
    #                     f"Please try to reduce the number of text system or "
    #                     f"assistant messages in the chat context."
    #                 )
    #
    #             self._text_response_retries += 1
    #             logger.warning(
    #                 "The OpenAI Realtime API returned a text response instead of audio. "
    #                 "Attempting to recover to audio mode...",
    #                 extra={
    #                     "item_id": message.item_id,
    #                     "text": message.text,
    #                     "retries": self._text_response_retries,
    #                 },
    #             )
    #             self._session._recover_from_text_response(message.item_id)
    #         else:
    #             self._text_response_retries = 0
    #
    #     @self._session.on("input_speech_committed")
    #     def _input_speech_committed():
    #         self._stt_forwarder.update(
    #             stt.SpeechEvent(
    #                 type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
    #                 alternatives=[stt.SpeechData(language="", text="")],
    #             )
    #         )
    #
    #     @self._session.on("input_speech_transcription_completed")
    #     def _input_speech_transcription_completed(ev: _InputTranscriptionProto):
    #         self._stt_forwarder.update(
    #             stt.SpeechEvent(
    #                 type=stt.SpeechEventType.FINAL_TRANSCRIPT,
    #                 alternatives=[stt.SpeechData(language="", text=ev.transcript)],
    #             )
    #         )
    #         if self._model.capabilities.supports_truncate:
    #             user_msg = ChatMessage.create(
    #                 text=ev.transcript, role="user", id=ev.item_id
    #             )
    #
    #             self._session._update_conversation_item_content(
    #                 ev.item_id, user_msg.content
    #             )
    #
    #         self._emit_speech_committed("user", ev.transcript)
    #
    #     @self._session.on("agent_speech_transcription_completed")
    #     def _agent_speech_transcription_completed(ev: _InputTranscriptionProto):
    #         self._agent_stt_forwarder.update(
    #             stt.SpeechEvent(
    #                 type=stt.SpeechEventType.FINAL_TRANSCRIPT,
    #                 alternatives=[stt.SpeechData(language="", text=ev.transcript)],
    #             )
    #         )
    #         self._emit_speech_committed("agent", ev.transcript)
    #
    #     # Similar to _input_speech_started, this handles updating the state to "listening" when the agent's speech is complete.
    #     # However, since Gemini doesn't support VAD events, we are not emitting the `user_started_speaking` event here.
    #     @self._session.on("agent_speech_stopped")
    #     def _agent_speech_stopped():
    #         self.interrupt()
    #
    #     @self._session.on("input_speech_started")
    #     def _input_speech_started():
    #         self.emit("user_started_speaking")
    #         self.interrupt()
    #
    #     @self._session.on("input_speech_stopped")
    #     def _input_speech_stopped():
    #         self.emit("user_stopped_speaking")
    #
    #     @self._session.on("function_calls_collected")
    #     def _function_calls_collected(fnc_call_infos: list[llm.FunctionCallInfo]):
    #         self.emit("function_calls_collected", fnc_call_infos)
    #
    #     @self._session.on("function_calls_finished")
    #     def _function_calls_finished(called_fncs: list[llm.CalledFunction]):
    #         self.emit("function_calls_finished", called_fncs)
    #
    #     @self._session.on("metrics_collected")
    #     def _metrics_collected(metrics: MultimodalLLMMetrics):
    #         self.emit("metrics_collected", metrics)