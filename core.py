from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel
from typing import List, Optional
import os
try:
    from cinesage.json_output import extract_json
except ModuleNotFoundError:
    from json_output import extract_json

class MovieInfo(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: Optional[List[str]] = None
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    rating: Optional[float] = None
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN not found in .env")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=token,
    max_new_tokens=2048,
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a movie information extraction assistant.

Extract the movie information from the user's text.

{format_instructions}
"""
    ),
    ("human", "{movie_information}")
])

para = input("Enter the movie information: ")

final_prompt = prompt_template.invoke({
    "movie_information": para,
    "format_instructions": parser.get_format_instructions()
})

result = model.invoke(final_prompt)

movie = parser.parse(extract_json(result.content))

print(movie.model_dump_json(indent=2))