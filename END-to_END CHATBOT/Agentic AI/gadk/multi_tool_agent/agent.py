import typing
from google.adk.agents import Agent

# The list of words for the word guessing game.
# We will put this directly in the agent's instructions.
WORD_LIST = ["dog", "cat", "elephant", "car"]

# ==============================================================================
#  TOOL FOR NUMBER GUESSING GAME
# ==============================================================================

def guess_number(
    min_val: int = 1,
    max_val: int = 100,
    last_guess: typing.Optional[int] = None,
    user_feedback: str = "start",
) -> dict:
    """
    Performs a binary search guess for the Number Guessing Game.

    To start the game, call this tool with no arguments.
    For subsequent turns, provide the agent's last guess and the user's
    feedback ('higher', 'lower', or 'correct').

    Args:
        min_val (int): The current minimum possible value for the number.
        max_val (int): The current maximum possible value for the number.
        last_guess (int): The agent's previous guess.
        user_feedback (str): The user's hint. Must be one of:
                             'start', 'higher', 'lower', 'correct'.

    Returns:
        A dictionary with the game status and the agent's next action or result.
    """
    # --- Update the search range based on user feedback ---
    if user_feedback.lower() == "higher":
        if last_guess is None:
            return {"status": "error", "message": "You must provide the 'last_guess' when feedback is 'higher'."}
        min_val = last_guess + 1
    elif user_feedback.lower() == "lower":
        if last_guess is None:
            return {"status": "error", "message": "You must provide the 'last_guess' when feedback is 'lower'."}
        max_val = last_guess - 1
    elif user_feedback.lower() == "correct":
        return {"status": "success", "message": f"Great! I guessed your number: {last_guess}!"}

    # --- Check for impossible situations ---
    if min_val > max_val:
        return {
            "status": "error",
            "message": "It looks like you gave me conflicting information! The search range is now empty. Let's start over.",
        }

    # --- Calculate the next guess (the middle of the new range) ---
    new_guess = (min_val + max_val) // 2

    return {
        "status": "continue",
        "message": f"My next guess is {new_guess}. Is your number higher, lower, or correct?",
        "new_guess": new_guess,
        "min_val": min_val,
        "max_val": max_val,
    }


# ==============================================================================
#  THE MAIN AGENT DEFINITION
# ==============================================================================

# Note: The Word Guessing game logic is handled by the instruction prompt,
# not by a separate tool. This leverages the LLM's reasoning capabilities.

root_agent = Agent(
    name="game_master_agent",
    model="gemini-1.5-flash",  # Using a modern, fast model
    description=(
        "A friendly agent that can play two games with the user: "
        "Number Guessing and Word Guessing."
    ),
    instruction=f"""
You are a fun and friendly Game Master. Your goal is to play guessing games with the user.
You can play two different games. Wait for the user to tell you which game they want to play.

--- GAME 1: Number Guessing ---
1. The user will think of a number between 1 and 100.
2. Your job is to guess the number using a binary search strategy.
3. To make a guess, you MUST use the `guess_number` tool.
4. To start the game, call the tool with its default values. The tool will return the first guess (50).
5. Present the guess to the user and ask if their number is "higher", "lower", or "correct".
6. Based on the user's response, call the `guess_number` tool again, providing the `last_guess` and the `user_feedback`.
7. Continue this process until the tool reports success or an error.

--- GAME 2: Word Guessing ---
1. The user will think of a word from this specific list: {WORD_LIST}.
2. Your job is to guess the word by asking smart, clarifying 'yes' or 'no' questions to eliminate possibilities.
3. DO NOT use a tool for this game. Use your own reasoning to narrow down the options from the list.
4. For example, you could ask "Is it an animal?" or "Is it a vehicle?".
5. Based on the user's answer, mentally remove the words that no longer fit.
6. When you are confident you have only one word left, make your final guess. For example: "Is your word 'cat'?"
""",
    tools=[guess_number],
)