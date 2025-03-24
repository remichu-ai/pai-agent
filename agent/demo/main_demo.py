#!/usr/bin/env python
from __future__ import annotations
import asyncio
import json
import logging
import uuid
import os
import wave
import numpy as np


from livekit import rtc
from livekit.agents import JobContext, AutoSubscribe

logger = logging.getLogger("demo-agent")
logger.setLevel(logging.INFO)


from dotenv import load_dotenv

load_dotenv()


# Audio parameters as per LiveKit docs.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1      # mono audio
SAMPLES_PER_CHANNEL = 480  # 10ms at 48kHz


def list_all_tools() -> dict[str, list[str] | str]:
    """
    Returns a dictionary listing all tools with keys 'active' and 'disabled'.
    If no tools are available in a category, returns "NONE" for that category.
    """
    active = ["get weather", "send email", "web search"]  # Dummy active tools
    disabled = []  # Dummy disabled tools
    return {
        "active": active if active else "NONE",
        "disabled": disabled if disabled else "NONE"
    }

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

async def broadcast_audio_file(ctx: JobContext, participant: rtc.Participant, file_name: str):
    """
    Broadcast a pre-recorded WAV audio file from the agent by creating an
    audio source and publishing its track.
    Additionally, publish the corresponding transcription file (assumed to have
    the same base name as the audio file but with a .txt extension) before the audio plays.
    """
    # Create an audio source and track.
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("pre-recorded-track", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    publication = await ctx.agent.publish_track(track, options)
    logger.info(f"Published audio track with SID: {publication.sid}")

    # Determine the correct track SID by checking the participant's track publications.
    track_sid = None
    for pub in participant.track_publications.values():
        if pub.source == rtc.TrackSource.SOURCE_MICROPHONE:
            track_sid = pub.sid
            break
    if not track_sid:
        track_sid = publication.sid  # fallback if no microphone track found

    # Publish the transcription before playing the audio.
    transcript_file = os.path.splitext(file_name)[0] + ".txt"
    try:
        with open(transcript_file, "r") as f:
            transcript_text = f.read()
        await send_transcription(
            ctx=ctx,
            participant=participant,
            track_sid=track_sid,
            segment_id=str(uuid.uuid4()),
            text=transcript_text,
            is_final=True,
        )
        logger.info(f"Published transcription for track SID: {track_sid}")
    except Exception as e:
        logger.error(f"Failed to publish transcription file '{transcript_file}': {e}")

    # Open the WAV file.
    try:
        wav_file = wave.open(file_name, 'rb')
    except Exception as e:
        logger.error(f"Failed to open WAV file '{file_name}': {e}")
        return

    # Check if WAV parameters match expected values.
    wav_rate = wav_file.getframerate()
    wav_channels = wav_file.getnchannels()
    if wav_rate != SAMPLE_RATE or wav_channels != NUM_CHANNELS:
        logger.warning("WAV file sample rate or channels do not match expected values "
                       f"(expected {SAMPLE_RATE}Hz/{NUM_CHANNELS}, got {wav_rate}Hz/{wav_channels}).")

    # Read and push frames until the end of the file.
    while True:
        frames = wav_file.readframes(SAMPLES_PER_CHANNEL)
        if not frames:
            break

        audio_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL)
        audio_data = np.frombuffer(audio_frame.data, dtype=np.int16)
        wav_array = np.frombuffer(frames, dtype=np.int16)
        np.copyto(audio_data[:len(wav_array)], wav_array)
        if len(wav_array) < SAMPLES_PER_CHANNEL:
            audio_data[len(wav_array):] = 0

        await source.capture_frame(audio_frame)
        await asyncio.sleep(SAMPLES_PER_CHANNEL / SAMPLE_RATE)

    logger.info("Finished playing the audio file.")



async def entrypoint(ctx: JobContext):
    logger.info("Connecting to room...")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("Waiting for participant...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")
    await broadcast_audio_file(ctx, participant, "hi.wav")
    # --- Mock RPC Methods ---

    @ctx.room.local_participant.register_rpc_method("createInitialResponse")
    async def create_initial_response(data: rtc.rpc.RpcInvocationData = None):
        logger.info("RPC 'createInitialResponse' invoked")
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("pg.updateConfig")
    async def update_config(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'pg.updateConfig' invoked with payload: " + data.payload)
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("interruptAgent")
    async def interrupt_agent(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'interruptAgent' invoked")
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("createResponse")
    async def create_response(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'createResponse' invoked")
        await broadcast_audio_file(ctx, participant, "answer.wav")
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("getToolList")
    async def get_tool_list(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'getToolList' invoked")
        tool_list = list_all_tools()
        return json.dumps(tool_list)

    @ctx.room.local_participant.register_rpc_method("setToolList")
    async def set_tool_list(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'setToolList' invoked with payload: " + data.payload)
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("setRecordStartTime")
    async def set_record_start_time(data: rtc.rpc.RpcInvocationData):
        if data.caller_identity != participant.identity:
            return
        logger.info("RPC 'setRecordStartTime' invoked with payload: " + data.payload)
        return json.dumps({"changed": "true"})

    @ctx.room.local_participant.register_rpc_method("clearHistory")
    async def clear_history(data: rtc.rpc.RpcInvocationData = None):
        logger.info("RPC 'clearHistory' invoked")
        return json.dumps({"changed": "true"})

    # Keep the script running.
    while True:
        await asyncio.sleep(1)


def run_livekit_worker():
    from livekit.agents.cli import cli
    from livekit.agents import WorkerOptions, WorkerType
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))


if __name__ == "__main__":
    run_livekit_worker()