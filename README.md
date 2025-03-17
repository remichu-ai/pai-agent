# PAI Agent

The accompany agent backend for PAI app.
This aim to be a simple agent backend that connect PAI app on phone to the gallama backend in order to perform voice chat.

The code here leverage Litkit Library to enable voice and video streaming via RTC.
The overall setup is as per following diagram from livekit

![livekit](https://github.com/remichu-ai/pai-agent/blob/main/image/agents-overview.svg)

## Components:
Following is the detail of each component
 - Backend LLM: gallama, running LLM, Text-to-Speech and Speech-to-Text 
 - Agent: `/agent` folder
 - Token service: `/token` folder *(Not mentioned in the diagram but is required for client to connect)*
 - Livekit Server: locally run Livekit server
 - Client: `PAI` app

## Download Model
### Native Gallama installation

I would suggest you try out the setup with: Whisper Large V3 Turbo + Qwen 2.5 VL 7B + Kokoro
- STT: whisper series is supported
- LLM: all LLM as supported by exllama v2 or llama cpp. 
- VLM: for vision capability, currently only support with exllama v2
  - Qwen VL 2.5 7B/ 70B
  - Pixtral (not as good as Qwen VL for video task)
- TTS:
  - Kokoro
  - GPT v2 with voice cloning (in progress)

Then you can change the model to something more to your need. 
If you are non-native Speaker, I would recommend to use Whisper Large V3 instead of the Turbo version

To see the list of model with automated downloader, you can run
```shell
gallama list available
```

Download model with gallama installed:
```shell
gallama download whisper-turbo
# gallama download whisper-large-v3

# You can change to another quantization as needed, but do use 8.0 if you can
gallama download qwen-2.5-VL-7B:8.0
```

### Using Docker

To enable GPUs usage in docker, please install NVIDIA Container Toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html


If you use docker, create a folder that will be where gallama read the model from e.g. 
`/home/yourid/gallama`

`yourid` here refer toy our your id in Linux

docker run -e GALLAMA_HOME_PATH=/home/remichu/gallama -it --network=host --gpus all -v /home/remichu/work/ML/gallama:/home/remichu/work/ML/gallama -v /home/remichu/gallama:/home/remichu/gallama -v /home/remichu/work/ML/model:/home/remichu/work/ML/model --entrypoint=/bin/bash gallama:dev


```shell
docker run -e GALLAMA_HOME_PATH=/home/yourid/gallama -it --network=host -v -v /home/yourid/gallama:/home/yourid/gallama remichu/gallama:latest /bash

# from here you would already be inside the gallama docker to run gallama command

gallama download whisper-large-v3-turbo
# gallama download whisper-large-v3

# You can change to another quantization as needed, but do use 8.0 if you can
gallama download qwen-2.5-VL-7B:8.0
```

## Run using Docker Compose
Running using Docker Compose is the easiest way to get start.
Ensure that you have:
- Docker installed (or any other container runtime tool with GPU support)
- Nvidia Container Toolkit installed

Download the model that you want to run and edit the docker compose accordingly *refer the docker-compose.yml*

Then run the following command:
```shell
docker compose up
```

## Run with Gallama installed natively

You can simply run the equivalent command as in the docker compose. Here are examples:
```shell
gallama run -id "model_name=whisper-turbo" -id "model_name=qwen-2.5-VL-7B max_seq_len=32768" -id "model_name=kokoro"
```

or with whisper large v3 and gpus setting:
```shell
gallama run -id "model_name=whisper-large-v3" -id "model_name=qwen-2.5-VL-7B gpus=0,16,21,21 max_seq_len=32768" -id "model_name=kokoro"
```

## Find your IP

THe last step you need is to find your IP to set inside PAI app so that it can connect to the backend.

### For local network access
You can find your server URL at by run this command on Ubuntu
`hostname -I`

### For remote access
For remote access, you will need a VPN service that let you create a private network between your phone and your backend.
I use tailscale (this is not endorsement), however, I would suggest that you do your research to find a solution that meet your security need. 