import openai
# Or create an OpenAI client with the gateway
client = openai.OpenAI(
    base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
    api_key="dummy",
    default_headers = {"X-Api-Key": "53de98c9d94276ae711822169237f3b8"}
)

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Hello from openai gateway!"}],
)

print(resp.choices[0].message.content)