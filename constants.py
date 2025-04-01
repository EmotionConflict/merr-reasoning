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

SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL = (
    """
    Imagine three different unimodal experts are analyzing the video, each focusing on a different modality:

    1. Expert A has access only to the transcript of the speech/dialogue.
    2. Expert B has access only to the audio track (tone, pitch, prosody, etc.).
    3. Expert C has access only to the visual cues (facial expressions, gestures, body language).

    All experts will write down 1 step of their thinking in turn, then share it with the group. Next, they all proceed to the next step, and so on. If any expert realizes they are incorrect at any point, they leave the discussion.

    After a sufficient exchange of ideas and deliberation, the experts must come to a final, single consensus.

    The question is:

    Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.
    """
)

SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL = (
    """
    Imagine three different experts are analyzing the video, each focusing on a different combination of modalities:

    1. Expert A has access to the text (transcript) and the visual content (facial expressions, gestures, etc.).
    2. Expert B has access to the text (transcript) and the audio track (tone, pitch, prosody, etc.).
    3. Expert C has access to the audio track and the visual content.

    All experts will write down 1 step of their thinking in turn and then share it with the group. Next, they move on to the next step, and so on. If any expert realizes they are incorrect at any point, they leave the discussion. After a sufficient exchange of ideas and deliberation, the experts must come to a single final conclusion.

    The question is...
    
    Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.
    """
)

SYSTEM_PROMPT_TREE_OF_THOUGHT_DEBATE = (
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