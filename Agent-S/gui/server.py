import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pyautogui
import io
import logging
import traceback

from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="gui"), name="static")

class Instruction(BaseModel):
    instruction: str
    model: str

@app.get("/")
async def read_root():
    try:
        return FileResponse("gui/index.html")
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error serving index page")

@app.get("/check-keys")
async def check_keys():
    try:
        return {
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "huggingface": bool(os.getenv("HF_TOKEN"))
        }
    except Exception as e:
        logger.error(f"Error checking API keys: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error checking API keys")

@app.post("/execute")
async def execute_instruction(instruction: Instruction):
    try:
        logger.info(f"Received instruction: {instruction.instruction}")
        logger.info(f"Using model: {instruction.model}")

        # Configure engine parameters based on selected model
        if "gemini" in instruction.model:
            engine_params = {
                "engine_type": "gemini",
                "model": instruction.model
            }
        else:
            if not os.getenv("HF_MODEL_ENDPOINT"):
                raise ValueError("HF_MODEL_ENDPOINT environment variable not set")
            engine_params = {
                "engine_type": "huggingface",
                "model": os.getenv("HF_MODEL_ENDPOINT")
            }

        # Initialize grounding agent
        logger.info("Initializing grounding agent...")
        try:
            grounding_agent = OSWorldACI(
                platform="linux",  # Assuming Linux for web environment
                engine_params_for_generation=engine_params,
                engine_params_for_grounding=engine_params
            )
        except Exception as e:
            logger.error(f"Error initializing grounding agent: {str(e)}\n{traceback.format_exc()}")
            raise

        # Initialize Agent-S
        logger.info("Initializing Agent-S...")
        try:
            agent = AgentS2(
                engine_params,
                grounding_agent,
                platform="linux",
                action_space="pyautogui",
                observation_type="mixed",
                search_engine="Perplexica"
            )
        except Exception as e:
            logger.error(f"Error initializing Agent-S: {str(e)}\n{traceback.format_exc()}")
            raise

        # Get screenshot
        logger.info("Capturing screenshot...")
        try:
            screenshot = pyautogui.screenshot()
            buffered = io.BytesIO()
            screenshot.save(buffered, format="PNG")
            screenshot_bytes = buffered.getvalue()
        except Exception as e:
            logger.error(f"Error capturing screenshot: {str(e)}\n{traceback.format_exc()}")
            raise

        # Execute instruction
        logger.info("Executing instruction...")
        try:
            info, action = agent.predict(
                instruction=instruction.instruction,
                observation={"screenshot": screenshot_bytes}
            )
        except Exception as e:
            logger.error(f"Error executing instruction: {str(e)}\n{traceback.format_exc()}")
            raise

        # Execute the action
        if action:
            logger.info(f"Executing action: {action[0]}")
            try:
                exec(action[0])
            except Exception as e:
                logger.error(f"Error executing action: {str(e)}\n{traceback.format_exc()}")
                raise

        response = f"Task completed. Info: {info}"
        logger.info(f"Task completed successfully: {response}")
        return {"response": response}

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
