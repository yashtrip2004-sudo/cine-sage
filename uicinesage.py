
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
import os
from pydantic import BaseModel
from typing import List, Optional
import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a movie information extraction assistant.

Your task is to analyze the movie-related information provided by the user and
extract the movie details into the required JSON structure.

Use only information supported by the user's input. Use null for optional
fields that cannot be determined.

Important rules:
- Extract information from the entire user input, even if it contains URLs,
  citations, Reddit comments, YouTube information, or unrelated formatting.
- Do not reproduce URLs.
- Do not explain your reasoning.
- Do not add information that cannot reasonably be inferred from the input.
Return only valid JSON. Do not include Markdown fences or explanations.

{format_instructions}
""",
        ),
        ("human", "{movie_information}"),
    ]
)


st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="centered"
)


st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #fff7fa 0%,
            #ffeaf2 50%,
            #fff8fb 100%
        );
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .title {
        text-align: center;
        color: #c94f7c;
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9d6279;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .input-title {
        color: #9d4268;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    textarea {
        background-color: #fffafd !important;
        border: 1px solid #efbfd1 !important;
        border-radius: 15px !important;
        color: #333333 !important;
    }

    textarea:focus {
        border: 1px solid #d96b94 !important;
        box-shadow: 0 0 0 1px #d96b94 !important;
    }

    .stButton > button {
        width: 100%;
        background: #d95f8c;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #c94f7c;
        color: white;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #f1c6d6;
        border-radius: 18px;
        padding: 22px;
        margin-top: 25px;
        box-shadow: 0 5px 18px rgba(210, 100, 140, 0.08);
    }

    .result-title {
        color: #000000;
        font-size: 22px;
        font-weight: 650;
        margin-bottom: 15px;
    }

    .result-content,
    .result-content p,
    .result-content strong,
    .result-content em,
    .result-content li,
    .result-content h1,
    .result-content h2,
    .result-content h3 {
        color: #000000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="title">🎬 CineSage</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover the essence of any movie</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-title">Movie Information</div>',
    unsafe_allow_html=True
)

movie_information = st.text_area(
    "Movie Information",
    placeholder="Paste movie information here...",
    height=280,
    label_visibility="collapsed"
)

if st.button("✨ Analyze Movie"):

    if movie_information.strip():

        final_prompt = prompt_template.invoke(
            {
                "movie_information": movie_information,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        try:
            with st.spinner("Analyzing movie..."):
                result = model.invoke(final_prompt)
                movie = parser.parse(extract_json(result.content))
        except (OutputParserException, ValueError) as error:
            st.error(f"Could not read structured JSON from the model response: {error}")
            st.stop()

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-title">🎞️ CineSage Analysis</div>',
            unsafe_allow_html=True
        )

        st.json(movie.model_dump())

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter movie information.")

