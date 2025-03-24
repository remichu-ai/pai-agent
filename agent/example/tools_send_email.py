import aiohttp
from typing import Annotated
from livekit.agents import llm
from DynamicFunctionContext import DynamicFunctionContext


import base64
from email.message import EmailMessage
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# first define a class that inherits from llm.FunctionContext
class AssistantFnc(DynamicFunctionContext):
    # the llm.ai_callable decorator marks this function as a tool available to the LLM
    # by default, it'll use the docstring as the function's description
    @llm.ai_callable()
    async def get_weather(
        self,
        # by using the Annotated type, arg description and type are available to the LLM
        location: Annotated[
            str, llm.TypeInfo(description="The location to get the weather for")
        ],
    ):
        """Called when the user asks about the weather. This function will return the weather for the given location."""
        # logger.info(f"getting weather for {location}")
        url = f"https://wttr.in/{location}?format=%C+%t"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    weather_data = await response.text()
                    # response from the function call is returned to the LLM
                    # as a tool response. The LLM's response will include this data
                    return f"The weather in {location} is {weather_data}."
                else:
                    raise f"Failed to get weather data, status code: {response.status}"

    @llm.ai_callable()
    async def send_email_to_self(
        self,
        # by using the Annotated type, arg description and type are available to the LLM
        title: Annotated[
            str, llm.TypeInfo(description="Email Title")
        ],
        content: Annotated[
            str, llm.TypeInfo(description="Email content to send to user own mailbox")
        ],
    ):
        """Send Email via Google API."""
        # you will need a credentials.json which will generate a token.json upon usage
        TOKEN_PATH = "token.json"
        CRED_PATH = "credentials.json"

        SCOPES = ['https://www.googleapis.com/auth/gmail.send']

        creds = None
        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CRED_PATH, SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        try:
            service = build("gmail", "v1", credentials=creds)
            message = EmailMessage()

            message.set_content(content)

            message["To"] = "myemail@gmail.com"
            message["From"] = "toemail@gmail.com"
            message["Subject"] = title

            # encoded message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            create_message = {"raw": encoded_message}
            # pylint: disable=E1101
            send_message = (
                service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute()
            )
            print(f'Message Id: {send_message["id"]}')
        except HttpError as error:
            print(f"An error occurred: {error}")
            send_message = None
            return "Failed to Send. Do not retry"
        return "Send Successfully"
