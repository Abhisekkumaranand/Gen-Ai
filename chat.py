from dotenv import load_dotenv

from groq import Groq

load_dotenv()

client=Groq()

result =client.chat.completions.create(
  model="llama-3.3-70b-versatile",
  messages=[
  
      {
            "role": "user",
            "content": "halo",
        }
  ]
)

print(result.choices[0].message.content)
# print(result) it is zero shot promting technique