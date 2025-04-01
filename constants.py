# This is what Emotion-LLaMA uses as the system prompt.
SYSTEM_PROMPT = (
    "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt. Answer the emotion label in lower-case."
)

# SYSTEM_PROMPT_BASELINE = (
#     '''
#     Act as an expert in emotion recognition. You are given a combination of transcript, audio cues, and visual cues.
#     - transcript: This field represents the transcript of what is said by the individual.
#     - audio_cues: This field contains a description of the audio characteristics—such as tone, intonation, pace, or voice quality—that give clues about the speaker’s emotion.
#     - visual_cues: This field is visual cues or observations, often related to facial expressions or body language. 
    
#     Your task is to determine the single dominant dominant emotion label the above information represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt. 
#     Answer the emotion label directly in lower-case.
#     '''
# )

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
