import json
import requests

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return f"Failed to fetch weather for {city}: {response.status_code}"


# Add more tools here later
tool_registry = {
    "get_weather": {
        "function": get_weather,
        "description": "Fetch current weather for a city using wttr.in. Input should be the city name.",
    },
}

available_tools = "\n".join(
    f"- {name}: {info['description']}" for name, info in tool_registry.items()
)

system_prompt = (
    "You are an assistant that can call tools when needed. "
    "When the user asks for something the assistant cannot answer directly, choose one tool and return JSON only. "
    "Use one of these exact JSON formats:\n"
    "{\"tool\": \"tool_name\", \"tool_input\": \"input string\"}\n"
    "or\n"
    "{\"response\": \"Your answer here\"}\n"
    "Available tools:\n"
    f"{available_tools}\n"
    "If the query is unrelated to tool-based operations, return a polite refusal in the `response` field. "
    "If the user asks for weather, call the `get_weather` tool with the city name."
)

messages = [
    {"role": "system", "content": system_prompt},
]
query = input("Enter your query: ")
messages.append({"role": "user", "content": query})

while True:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=messages,
    )

    assistant_text = response.choices[0].message.content
    try:
        assistant_json = json.loads(assistant_text)
    except json.JSONDecodeError:
        print("Model did not return valid JSON.\n", assistant_text)
        break

    if "tool" in assistant_json:
        tool_name = assistant_json["tool"]
        tool_input = assistant_json.get("tool_input", "")

        if tool_name not in tool_registry:
            print(f"Unknown tool requested: {tool_name}")
            break

        tool_func = tool_registry[tool_name]["function"]
        tool_result = tool_func(tool_input)
        print(f"Tool `{tool_name}` result:\n{tool_result}\n")

        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "name": tool_name, "content": tool_result})
        continue

    if "response" in assistant_json:
        print(assistant_json["response"])
        break

    print("Unexpected response format:", assistant_json)
    break