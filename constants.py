# This is what Emotion-LLaMA uses as the system prompt.
SYSTEM_PROMPT = (
    "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt. Answer the emotion label in lower-case."
)

SYSTEM_PROMPT_CHAIN_OF_THOUGHT = (
    """
    Let's think step by step.
    Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.
    """
)

SYSTEM_PROMPT_TREE_OF_THOUGHT = (
    """
    Imagine three different experts are answering this question.
    All experts will write down 1 step of their thinking,
    then share it with the group.
    Then all experts will go on to the next step, etc.
    If any expert realises they're wrong at any point then they leave.
    The question is...
    
    Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.
    """
)
