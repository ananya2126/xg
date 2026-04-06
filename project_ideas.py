from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os
from groq import Groq

# Initialize Groq client once
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_project_ideas(resume_text, skills):

    if not resume_text:
        return "Resume text is required."

    try:

        skill_list = ", ".join(skills)

        prompt = f"""
Based on the following resume and skills, suggest 3 impactful real-world project ideas.

Requirements:
- Must solve real-world problems
- Must align with candidate skills
- Must impress recruiters
- Provide:
  • Project title
  • Description
  • Technologies used
  • Expected outcome

Resume:
{resume_text}

Skills:
{skill_list}
"""


        completion = client.chat.completions.create(

            model="groq/compound",

            messages=[

                {
                    "role": "system",
                    "content": """
                        You are a software architect.

                        Return ONLY ONE project idea in markdown format.

                        STRICT FORMAT:

                        ## Project Title: <Title>

                        **Description:** <text>

                        **Technologies:** <text>

                        **Outcome:** <text>

                        Do NOT generate multiple projects.
                        Do NOT add introduction.
                        Do NOT add conclusion.
                        Return ONLY the project.
                        """
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0.7,

            max_completion_tokens=1024,

            compound_custom={

                "tools": {

                    "enabled_tools": [

                        "web_search",
                        "code_interpreter",
                        "visit_website"

                    ]

                }

            }

        )


        return completion.choices[0].message.content


    except Exception as e:

        print("Compound AI error:", str(e))

        return "Unable to generate project ideas at this time."
