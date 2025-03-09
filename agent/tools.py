import aiohttp
from typing import Annotated

from livekit.agents import llm


class DynamicFunctionContext(llm.FunctionContext):
    """
    FunctionContext class with ability to toggle on and off tool
    """
    def __init__(self):
        super().__init__()
        # Store disabled tools here so that we can re-enable them later.
        self._disabled_tools: dict[str, llm.FunctionInfo] = {}

    def disable_tool(self, tool_name: str) -> bool:
        """
        Disables a tool by removing it from the active tools.
        Returns True if the tool was successfully disabled, otherwise False.
        """
        if tool_name in self._fncs:
            # Move the tool from active functions to the disabled dictionary.
            self._disabled_tools[tool_name] = self._fncs.pop(tool_name)
            return True
        return False

    def enable_tool(self, tool_name: str) -> bool:
        """
        Enables a previously disabled tool by moving it back to the active tools.
        Returns True if the tool was successfully re-enabled, otherwise False.
        """
        if tool_name in self._disabled_tools:
            self._fncs[tool_name] = self._disabled_tools.pop(tool_name)
            return True
        return False

    def list_active_tools(self) -> list[str]:
        """Returns a list of names for all currently active tools."""
        return list(self._fncs.keys())

    def list_disabled_tools(self) -> list[str]:
        """Returns a list of names for all currently disabled tools."""
        return list(self._disabled_tools.keys())

    def list_all_tools(self) -> dict[str, list[str] | str]:
        """
        Returns a dictionary listing all tools with keys 'active' and 'disabled'.
        If no tools are available in a category, returns "NONE" for that category.
        """
        active = self.list_active_tools()
        disabled = self.list_disabled_tools()

        return {
            "active": active if active else "NONE",
            "disabled": disabled if disabled else "NONE"
        }

    def disable_all_tools(self) -> None:
        """
        Disables all tools by moving all active tools to the disabled tools dictionary.
        """
        # Move all active tools to the disabled dictionary.
        self._disabled_tools.update(self._fncs)
        self._fncs.clear()

    def enable_all_tools(self) -> None:
        """
        Enables all tools by moving all disabled tools back to the active tools dictionary.
        """
        # Move all disabled tools back to the active functions.
        self._fncs.update(self._disabled_tools)
        self._disabled_tools.clear()

    def set_tool(self, tool_names: list[str]) -> None:
        """
        Enables only the tools specified in tool_names and disables any tool not in the list.

        Args:
            tool_names (list[str]): List of tool names to be enabled.
        """
        target_set = set(tool_names)

        # Disable active tools that are not in the target set.
        for tool in list(self._fncs.keys()):
            if tool not in target_set:
                self.disable_tool(tool)

        # Enable disabled tools that are in the target set.
        for tool in list(self._disabled_tools.keys()):
            if tool in target_set:
                self.enable_tool(tool)


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