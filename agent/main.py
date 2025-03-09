from __future__ import annotations
import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Union, List
import requests


from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from livekit.plugins.openai._oai_api import build_oai_function_description
from livekit.plugins.openai.realtime import api_proto


from livekit import api
from data_model import GallamaSessionConfig, TurnDetectionConfig
import functools

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

from MultiModalAgentModified import MultimodalAgentModified
from livekit.agents import llm, transcription, vad
from typing import Literal, Protocol
import os


try:
    from tools import AssistantFnc
    assistantFnc = AssistantFnc()
except ImportError as e:
    logger.info(f"Failed to import {AssistantFnc.__name__}: {e}")
    AssistantFnc = None
    assistantFnc = None


# the list of allowed tool for function calling
# Possible values are:
# True -> allow all available tools
# False -> disable tool
# or list of function name allowed

# allowed_tools: Union[bool, List[str]] = True

last_config: Dict[str, Any] = {}
assistant: MultimodalAgent = None

with open("system_prompt.txt", "r") as file:
    DEFAULT_INSTRUCTION = file.read()
#
# DEFAULT_INSTRUCTION = """
# Remember these key points:
# - Keep responses brief, clear, and natural-sounding (like casual speech)
# - Prefer simple words over complex terminology
# - Break longer responses into shorter sentences
#
# Speech Recognition Handling:
# - User input comes from speech transcription and may contain errors
# - Common issues include pronunciation-based mistakes from non-native English speakers
# - If a word seems incorrect or unclear, politely ask for clarification like "Did you mean [likely word]?"
# - When in doubt about the entire question, confirm by repeating it back
#
# Interactive Elements:
# - User may interrupt during your response with <user_interrupt>
# - When interrupted, stop immediately and address the interruption
# - Resume previous topic only if explicitly requested
#
# Response Style:
# - Your response will be converted to voice to playback to user by TTS.
# - When asking for how to pronounce words, phonetic spellings and romanization are prohibited. Example:
#   Allow: 猫比狗聪明
#   PROHIBITED: 猫比狗聪明 (máo bǐ gǒu cōng míng)
# - Use conversational markers like "Well," "You know," "Right"
# - Express agreement/understanding with brief acknowledgments
# - Mirror the user's speaking pace and energy level
#
# Video:
# - User might send you video from his phone for live question answer
# - The video might be shared with you in the form of a series of consecutive images
#
# Overall:
# Be a friendly, upbeat and concise conversational assistant.
# You are LLM in an STT to LLM to TTS system.
# Keep your initial greeting to user to 1 sentence.
# """


@dataclass
class SessionConfig:
    openai_api_key: str
    instructions: str
    voice: openai.realtime.api_proto.Voice
    temperature: float
    max_response_output_tokens: str | int
    modalities: list[openai.realtime.api_proto.Modality]
    turn_detection: openai.realtime.ServerVadOptions

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = self._modalities_from_string("text_and_audio")

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "openai_api_key"}

    @staticmethod
    def _modalities_from_string(modalities: str) -> list[str]:
        modalities_map = {
            "text_and_audio": ["text", "audio"],
            "text_only": ["text"],
        }
        return modalities_map.get(modalities, ["text", "audio"])

    def __eq__(self, other: SessionConfig) -> bool:
        return self.to_dict() == other.to_dict()



from copy import deepcopy
_original_session_update = openai.realtime.RealtimeSession.session_update

@functools.wraps(_original_session_update)
def session_update_gallama(self, *args, **kwargs):

    # get the keywords argument from orginal livekits implementation from kwarg
    modalities = kwargs.get("modalities", None)
    instructions = kwargs.get("instructions", None)
    voice = kwargs.get("voice", None)
    input_audio_format = kwargs.get("input_audio_format", None)
    output_audio_format = kwargs.get("output_audio_format", None)
    input_audio_transcription = kwargs.get("input_audio_transcription", None)
    turn_detection = kwargs.get("turn_detection", TurnDetectionConfig())
    tool_choice = kwargs.get("tool_choice", None)
    temperature = kwargs.get("temperature", None)
    max_response_output_tokens = kwargs.get("max_response_output_tokens", 4096)

    # convert turn_detection to pydantic
    if isinstance(turn_detection, dict):
        turn_detection = TurnDetectionConfig(**turn_detection)

    self._opts = deepcopy(self._opts)
    if modalities is not None:
        self._opts.modalities = modalities
    if instructions is not None:
        self._opts.instructions = instructions
    if voice is not None:
        self._opts.voice = voice
    if input_audio_format is not None:
        self._opts.input_audio_format = input_audio_format
    if output_audio_format is not None:
        self._opts.output_audio_format = output_audio_format
    if input_audio_transcription is not None:
        self._opts.input_audio_transcription = input_audio_transcription
    if turn_detection is not None and turn_detection.create_response:
        self._opts.turn_detection = turn_detection
    else:
        self._opts.turn_detection = None
    if tool_choice is not None:
        self._opts.tool_choice = tool_choice
    if temperature is not None:
        self._opts.temperature = temperature
    if max_response_output_tokens is not None:
        self._opts.max_response_output_tokens = max_response_output_tokens

    tools = []
    if self._fnc_ctx is not None:
        # for fnc in self._fnc_ctx.ai_functions:
        #     logger.info(f"{fnc}")

        for fnc in self._fnc_ctx.ai_functions.values():
            # the realtime API is using internally-tagged polymorphism.
            # build_oai_function_description was built for the ChatCompletion API
            # if allowed_tools is True or (isinstance(allowed_tools, str) and fnc.name in allowed_tools):

            function_data = build_oai_function_description(fnc)["function"]
            function_data["type"] = "function"
            tools.append(function_data)

    server_vad_opts: api_proto.ServerVad | None = None
    if self._opts.turn_detection is not None:
        server_vad_opts = {
            "type": "server_vad",
            "threshold": self._opts.turn_detection.threshold,
            "prefix_padding_ms": self._opts.turn_detection.prefix_padding_ms,
            "silence_duration_ms": self._opts.turn_detection.silence_duration_ms,
        }
    input_audio_transcription_opts: api_proto.InputAudioTranscription | None = None
    if self._opts.input_audio_transcription is not None:
        input_audio_transcription_opts = {
            "model": self._opts.input_audio_transcription.model,
        }

    session_data: api_proto.ClientEvent.SessionUpdateData = {
        "modalities": self._opts.modalities,
        "instructions": self._opts.instructions,
        "voice": self._opts.voice,
        "input_audio_format": self._opts.input_audio_format,
        "output_audio_format": self._opts.output_audio_format,
        "input_audio_transcription": input_audio_transcription_opts,
        "turn_detection": server_vad_opts,
        "tools": tools,
        "tool_choice": self._opts.tool_choice,
        "temperature": self._opts.temperature,
        "max_response_output_tokens": None,
    }

    # azure doesn't support inf for max_response_output_tokens
    if not self._opts.is_azure or isinstance(
        self._opts.max_response_output_tokens, int
    ):
        session_data["max_response_output_tokens"] = (
            self._opts.max_response_output_tokens
        )
    else:
        del session_data["max_response_output_tokens"]  # type: ignore

    # merge config
    global last_config
    session_data = GallamaSessionConfig(
        **{
            **last_config,
            **kwargs,
            **session_data
        }
    ).model_dump(exclude_unset=False)

    # update turn detection
    global assistant

    if session_data.get("turn_detection"):
        model = openai.realtime.RealtimeModel(
            api_key=os.getenv('OPENAI_API_KEY', 'dev'),
            instructions=session_data.get("instructions"),
            voice=session_data.get("voice"),
            temperature=session_data.get("temperature"),
            max_response_output_tokens=session_data.get("max_response_output_tokens"),
            modalities=session_data.get("modalities"),
            turn_detection=TurnDetectionConfig(**session_data.get("turn_detection")),
        )
        if session_data.get("turn_detection").get("create_response"):
            if assistant.vad:
                # already have vad on
                pass
            else:
                # replace model
                assistant._model = model
                assistant._vad = TurnDetectionConfig(**session_data.get("turn_detection")),
        else:   # disable VAD
            assistant._model = model
            assistant._vad = None,

    logger.info(f"Updated session data: {session_data}")

    # track config
    last_config = session_data

    self._queue_msg(
        {
            "type": "session.update",
            "session": session_data,
        }
    )


# overwrite session update of livekits
openai.realtime.RealtimeSession.session_update = session_update_gallama


#########################

def parse_session_config(data: Dict[str, Any]) -> SessionConfig:
    turn_detection = None

    if data.get("turn_detection"):
        turn_detection_json = json.loads(data.get("turn_detection"))
        turn_detection = openai.realtime.ServerVadOptions(
            threshold=turn_detection_json.get("threshold", 0.5),
            prefix_padding_ms=turn_detection_json.get("prefix_padding_ms", 200),
            silence_duration_ms=turn_detection_json.get("silence_duration_ms", 300),
        )
    else:
        turn_detection = openai.realtime.DEFAULT_SERVER_VAD_OPTIONS

    config = SessionConfig(
        openai_api_key=data.get("openai_api_key", ""),
        instructions=data.get("instructions", ""),
        voice=data.get("voice", "alloy"),
        temperature=float(data.get("temperature", 0.8)),
        max_response_output_tokens=data.get("max_output_tokens")
        if data.get("max_output_tokens") == "inf"
        else int(data.get("max_output_tokens") or 2048),
        modalities=SessionConfig._modalities_from_string(
            data.get("modalities", "text_and_audio")
        ),
        turn_detection=turn_detection,
    )
    return config

def parse_session_config_gallama(data: Dict[str, Any]) -> GallamaSessionConfig:
    turn_detection = None
    logger.info(f"Parsing session config Gallama: {data}")

    if data.get("turn_detection"):
        if isinstance(data.get("turn_detection"), str):
            turn_detection_json = json.loads(data.get("turn_detection"))
        else:
            turn_detection_json = data.get("turn_detection")

    # set turn detection data to the data object
        data["turn_detection"] = TurnDetectionConfig(**turn_detection_json).model_dump()

    # handle how livekit pass modalities
    if not data.get("modalities", None):
        data["modalities"] = ["text", "audio"]
    elif data.get("modalities") == "text_and_audio":
        data["modalities"] = ["text", "audio"]

    # use default instruction if it is empty
    if not data.get("instructions", None):
        data["instructions"] = DEFAULT_INSTRUCTION

    # tool from front end will only containt the name
    # handling of tool alrd handled in setToolList and getToolList
    # remove tool to not overwrite the setting in the session update
    if data.get("tools"):
        data.pop("tools")

    config = GallamaSessionConfig(**data)
    return config


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    # await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")



def send_livekit_info_to_backend():
    """
    This function send livekit token to gallama server backend for it to connect to live kits room for video subscription
    """
    livekit_url = os.getenv('LIVEKIT_URL', 'ws://127.0.0.1:7880')
    openai_base_url = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')

    token = api.AccessToken(os.getenv('LIVEKIT_API_KEY', 'devkey'), os.getenv('LIVEKIT_API_SECRET', 'secret')) \
        .with_identity("backend") \
        .with_name("backend") \
        .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
    ))
    # send to video endpoint this payload to openai_base_url
    # Prepare the payload
    payload = {
        "token": token.to_jwt(),
        "livekit_url": livekit_url,
    }

    # Send the payload to the backend via an HTTP POST request
    try:
        response = requests.post(
            f"{openai_base_url}/video_via_livekit",  # Endpoint for the backend
            json=payload,  # Send the payload as JSON
            headers={"Content-Type": "application/json"}  # Set the content type
        )

        # Check if the request was successful
        if response.status_code == 200:
            print("LiveKit info successfully sent to the backend.")
        else:
            print(
                f"Failed to send LiveKit info to the backend. Status code: {response.status_code}, Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Error sending LiveKit info to the backend: {e}")


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    # setup video
    send_livekit_info_to_backend()
    global assistant

    try:
        metadata = json.loads(participant.metadata)
        config = parse_session_config(metadata)
    except Exception as e:
        config = GallamaSessionConfig()


    model = openai.realtime.RealtimeModel(
        api_key=os.getenv('OPENAI_API_KEY', 'dev'),
        instructions=config.instructions,
        voice=config.voice,
        temperature=config.temperature,
        max_response_output_tokens=config.max_response_output_tokens,
        modalities=config.modalities,
        turn_detection=config.turn_detection,
    )
    # assistant = MultimodalAgentModified(model=model)
    assistant = MultimodalAgentModified(
        model=model,
        fnc_ctx=assistantFnc    # tool usage
    )
    assistant.start(ctx.room)
    session = model.sessions[0]

    if config.modalities == ["text", "audio"]:
        session.conversation.item.create(
            llm.ChatMessage(
                role="user",
                content="Hello",
            )
        )
        session.response.create()


    # modified method for gallama
    @ctx.room.local_participant.register_rpc_method("pg.updateConfig")
    async def update_config(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        The standard updateConfig above by livekit example implementations follows OpenAI API
        Gallama however provide additional setting that user can set hence this method

        :param data:
            has a payload that is loadable into a GallamaSessionConfig
        """
        if data.caller_identity != participant.identity:
            return

        # new config for gallama
        new_config = parse_session_config_gallama(json.loads(data.payload))
        logger.info(f"new_config: {new_config}")

        if config != new_config:
            logger.info(
                f"config changed: {new_config.to_dict()}, participant: {participant.identity}"
            )
            session = model.sessions[0]
            session.session_update(
                **new_config.model_dump(exclude_unset=False)
            )

            # disable tool if needed:
            if not new_config.tools:
                assistant.fnc_ctx = None
            else:
                assistant.fnc_ctx = AssistantFnc()

            return json.dumps({"changed": "true"})
        else:
            return json.dumps({"changed": "false"})

    @ctx.room.local_participant.register_rpc_method("interruptAgent")
    async def interrupt_agent(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        Interrupt current agent speaking
        """
        if data.caller_identity != participant.identity:
            return

        global assistant

        try:
            # new config for gallama
            logger.info("interrupt agent")

            assistant.interrupt()

            return json.dumps({"changed": "true"})
        except Exception as err:
            logger.error(err)
            return json.dumps({"changed": "false"})

    @ctx.room.local_participant.register_rpc_method("createResponse")
    async def create_response(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        Trigger response creation. To be used when VAD not activated
        """
        if data.caller_identity != participant.identity:
            return

        global assistant

        try:
            # new config for gallama
            logger.info("response create")
            assistant.generate_reply(on_duplicate="cancel_existing", enable_vad=False)

            return json.dumps({"changed": "true"})
        except Exception as err:
            logger.error(err)
            return json.dumps({"changed": "false"})


    @ctx.room.local_participant.register_rpc_method("getToolList")
    async def get_tool_list(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        This function return a list of available tool for front end to select

        :param data:
            a list of available tool name
        """
        if data.caller_identity != participant.identity:
            return



        global assistant
        global assistantFnc
        tool_list = assistant.fnc_ctx
        if tool_list:
            tool_list = tool_list.list_all_tools()
        elif tool_list is None and assistantFnc:
            tool_list = assistantFnc.list_all_tools()

        logger.info(f"current tool_list: {tool_list}")
        return json.dumps(tool_list)

        # # new config for gallama
        # fnc_ctx = model.sessions[0].fnc_ctx
        #
        # tool_list = []
        # if fnc_ctx is not None:
        #     for fnc in fnc_ctx.ai_functions.values():
        #         # the realtime API is using internally-tagged polymorphism.
        #         # build_oai_function_description was built for the ChatCompletion API
        #         tool_list.append(fnc.name)
        #
        # if not tool_list:
        #     tool_list = "NONE"
        # return json.dumps({"tool_list": tool_list})


    @ctx.room.local_participant.register_rpc_method("setToolList")
    async def set_tool_list(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        This function return a list of available tool for front end to select

        :param data:
            a list of available tool name
        """
        if data.caller_identity != participant.identity:
            return


        logger.info(f"data.payload: {data.payload}")
        tool_list: Union[bool, List[str]] = json.loads(data.payload).get("tool_list", False)
        logger.info(f"set tool_list: {tool_list}")

        global assistantFnc
        global assistant
        assistantFnc = AssistantFnc()
        assistantFnc.set_tool(tool_list)
        assistant.fnc_ctx = assistantFnc

        session = model.sessions[0]
        session.session_update()


        #
        # global allowed_tools
        # allowed_tools = tool_list

        # return json.dumps({"changed": "true", "tool_list": tool_list})
        return json.dumps({"changed": "true"})


    @ctx.room.local_participant.register_rpc_method("setRecordStartTime")
    async def set_record_start_time(
        data: rtc.rpc.RpcInvocationData,
    ):
        """
        This function send the time stamp when user click the record button on non handfree mode

        :param data:
            a list of available tool name
        """
        if data.caller_identity != participant.identity:
            return


        logger.info(f"data.payload: {data.payload}")
        start_time: Union[bool, List[str]] = json.loads(data.payload).get("start_time", False)
        logger.info(f"start time: {start_time}")

        openai_base_url = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')


        # send to video endpoint this payload to openai_base_url
        # Prepare the payload
        payload = {
            "start_time": start_time,
        }

        # Send the payload to the backend via an HTTP POST request
        try:
            response = requests.post(
                f"{openai_base_url}/record_start_time",  # Endpoint for the backend
                json=payload,  # Send the payload as JSON
                headers={"Content-Type": "application/json"}  # Set the content type
            )

            # Check if the request was successful
            if response.status_code == 200:
                logger.debug("Successfully sent record start time to the backend.")
            else:
                print(
                    f"Failed to send record start time to the backend. Status code: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to send record start time to the backend: {e}")

        return json.dumps({"changed": "true"})


    @session.on("response_done")
    def on_response_done(response: openai.realtime.RealtimeResponse):
        variant: Literal["warning", "destructive"]
        description: str | None = None
        title: str
        if response.status == "incomplete":
            if response.status_details and response.status_details["reason"]:
                reason = response.status_details["reason"]
                if reason == "max_output_tokens":
                    variant = "warning"
                    title = "Max output tokens reached"
                    description = "Response may be incomplete"
                elif reason == "content_filter":
                    variant = "warning"
                    title = "Content filter applied"
                    description = "Response may be incomplete"
                else:
                    variant = "warning"
                    title = "Response incomplete"
            else:
                variant = "warning"
                title = "Response incomplete"
        elif response.status == "failed":
            if response.status_details and response.status_details["error"]:
                error_code = response.status_details["error"]["code"]
                if error_code == "server_error":
                    variant = "destructive"
                    title = "Server error"
                elif error_code == "rate_limit_exceeded":
                    variant = "destructive"
                    title = "Rate limit exceeded"
                else:
                    variant = "destructive"
                    title = "Response failed"
            else:
                variant = "destructive"
                title = "Response failed"
        else:
            return

        asyncio.create_task(show_toast(title, description, variant))

    async def send_transcription(
        ctx: JobContext,
        participant: rtc.Participant,
        track_sid: str,
        segment_id: str,
        text: str,
        is_final: bool = True,
    ):
        transcription = rtc.Transcription(
            participant_identity=participant.identity,
            track_sid=track_sid,
            segments=[
                rtc.TranscriptionSegment(
                    id=segment_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    language="en",
                    final=is_final,
                )
            ],
        )
        await ctx.room.local_participant.publish_transcription(transcription)

    async def show_toast(
        title: str,
        description: str | None,
        variant: Literal["default", "success", "warning", "destructive"],
    ):
        await ctx.room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="pg.toast",
            payload=json.dumps(
                {"title": title, "description": description, "variant": variant}
            ),
        )

    last_transcript_id = None

    # send three dots when the user starts talking. will be cleared later when a real transcription is sent.
    @session.on("input_speech_started")
    def on_input_speech_started():
        nonlocal last_transcript_id
        remote_participant = next(iter(ctx.room.remote_participants.values()), None)
        if not remote_participant:
            return

        track_sid = next(
            (
                track.sid
                for track in remote_participant.track_publications.values()
                if track.source == rtc.TrackSource.SOURCE_MICROPHONE
            ),
            None,
        )
        if last_transcript_id:
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )

        new_id = str(uuid.uuid4())
        last_transcript_id = new_id
        asyncio.create_task(
            send_transcription(
                ctx, remote_participant, track_sid, new_id, "…", is_final=False
            )
        )

    @session.on("input_speech_transcription_completed")
    def on_input_speech_transcription_completed(
        event: openai.realtime.InputTranscriptionCompleted,
    ):
        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )
            last_transcript_id = None

    @session.on("input_speech_transcription_failed")
    def on_input_speech_transcription_failed(
        event: openai.realtime.InputTranscriptionFailed,
    ):
        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )

            error_message = "⚠️ Transcription failed"
            asyncio.create_task(
                send_transcription(
                    ctx,
                    remote_participant,
                    track_sid,
                    last_transcript_id,
                    error_message,
                )
            )
            last_transcript_id = None

def run_livekit_worker():
    """Run the LiveKit worker."""
    from livekit.agents.cli import cli
    from livekit.agents import WorkerOptions, WorkerType
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))

if __name__ == "__main__":

    # Start the LiveKit worker in the main process
    run_livekit_worker()
