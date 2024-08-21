import instructor
from pydantic import BaseModel, Field
# from openai import OpenAI
from langfuse.openai import OpenAI
import dotenv
from typing import Dict
import langfuse


# Define your desired output structure
class UserInfo(BaseModel):
    name_to_age: dict[str, int] = Field(
            description="The users name and age")


# Patch the OpenAI client
dotenv.load_dotenv()
client = instructor.patch(OpenAI())

print(UserInfo.model_json_schema())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserInfo,
    messages=[
        {"role": "system", "content": "Please provide the name and ages of the users as a dictionary."},
        {"role": "user", "content": "John Doe is 30 years old. Anne Smith is 25 years old."}],
)

print(user_info.name_to_age)
#> John Doe
