from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
from enum import Enum


_LANGUAGE_CODES = [
    "auto",
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
]

# Create a reusable type for language
LanguageType = Optional[Union[List[Literal[*_LANGUAGE_CODES]], Literal[*_LANGUAGE_CODES]]]

class AudioFormat(str, Enum):
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"

class AudioTranscriptionConfig(BaseModel):
    model: Literal["whisper-1"] = "whisper-1"

class ToolParameter(BaseModel):
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]


class Tool(BaseModel):
    type: Literal["function"]
    name: str
    description: str
    parameters: ToolParameter

class TurnDetectionConfig(BaseModel):
    type: Literal["server_vad"] = "server_vad"
    threshold: Optional[float] = Field(ge=0.0, le=1.0,default=0.5)
    prefix_padding_ms: Optional[int] = Field(ge=0, default=300)
    silence_duration_ms: Optional[int] = Field(ge=0, default=400)
    create_response: bool = True

    # gallama specific setting
    language: Optional[LanguageType] = ["auto"]
    factor_prefix_padding_in_truncate: bool = Field(default=True,
                                                    description="Prefix padding will ensure speech start event only emitted "
                                                                "after certain ms of continuous speak, after which user will send conversation.item.truncate event"
                                                                "This setting is to automatically offset truncate timing by this amount of ms")


class VideoStreamSetting(BaseModel):
    video_stream: Optional[bool] = True
    # if video_max_resolution is None, there is no rescaling of image
    video_max_resolution: Literal["240p", "360p", "480p", "540p", "720p", "900p", "1080p", None] = "720p"
    retain_video: Optional[Literal["disable","message_based", "time_based"]] = Field(
        description="whether to retain images for past message", default="time_based")
    retain_per_message: int = Field(
        description="number of frame retained per message for old messages", default=1)
    second_per_retain: int = Field(
        description="one frame will be retained per this number of seconds", default=3)
    max_message_with_retained_video: int = Field(
        description="number of User messages that will have video frame retained", default=10)

class GallamaSessionConfig(BaseModel):
    modalities: List[Literal["text", "audio"]] = Field(default_factory=lambda: ["text", "audio"])
    instructions: Optional[str] = ""    # system prompt
    voice: Optional[str] = None
    input_audio_format: Optional[AudioFormat] = None
    output_audio_format: Optional[AudioFormat] = None
    input_audio_transcription: Optional[AudioTranscriptionConfig] = None
    turn_detection: Optional[TurnDetectionConfig] = Field(default_factory=TurnDetectionConfig)
    tools: Optional[Union[List[Tool], None]] = Field(default_factory=list)
    tool_choice: Optional[Literal["auto", "none", "required"]] = "auto"
    temperature: Optional[float] = Field(0.4, ge=0.1, le=1.2)
    max_response_output_tokens: Optional[Union[int, Literal["inf"]]] = "inf"

    # extra
    video: VideoStreamSetting = Field(default_factory=VideoStreamSetting)
    model: Optional[str] = None

    # extra for gallama backend
    streaming_transcription: bool = True
    user_interrupt_token: Optional[str] = Field(description= "Custom word to insert everytime user interrupt the assistant",default=" <user_interrupt>")
    input_sample_rate: Optional[int] = Field(description="Sample rate of input audio",default=24000)
    output_sample_rate: Optional[int] = Field(description="Sample rate of input audio",default=24000)

    # extra argument for gallama tool calling:
    tool_call_thinking: bool = Field(default= False, description="Automatically trigger one liner tool call thinking when tool in auto mode to decide if tool is required")
    tool_call_thinking_token: int = Field(default= 200, description="Maximum token for tool thinking generation. If it exceed this threshold, no tool thinking is returned")
    tool_instruction_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the general instruction to use tool. prefix for best kv caching"))
    tool_schema_position: Literal["prefix", "postfix"] = (
        Field(default="prefix", description="Position of the schema of individual tools. If tool_schema is unchanged through out, "
                                            "keep it as prefix for maximum kv caching. postfix for cases where tool are changing between api request"))

    def to_dict(self):
        return self.model_dump(exclude_unset=True)

    # class Config:
    #     extra = "allow"  # Allow extra fields


