SYSTEM_PROMPT = (
   """
You are given a multimodal_description extracted from a video. Your task is to identify the dominant emotion among [disgust, surprise, anger, joy, fear, sadness, neutral] being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported. Please base your judgment on the input provided and respond according to the final_output format below. 

IMPORTANT NOTES: 
For emotion labels, Insert one of: [disgust, surprise, anger, joy, fear, sadness, neutral] in lowercase only without any other text.

Based on the input provided, please provide final_output in the following format: 
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1-100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1-100 (leave blank if no second_emotion)",
  "justification": "Provide a brief reasoning (1-2 sentences) for your chosen emotions."
}
   """
)


##################


SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT_DEBATE = (
"""
You are given a multimodal_description extracted from a video. Please provide a response in JSON format only, without any other text. Your task is to identify the dominant emotion being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

Three unimodal experts are analyzing the video, each focusing on a different modality:

  1. Expert in Text Interpretation (A) has access only to: \"whisper_transcript\"
  2. Expert in Audio Interpretation (B) has access only to: \"audio_description\"
  3. Expert in Visual Interpretation (C) has access only to: \"visual_expression_description\" and \"visual_objective_description\"

All experts must first identify the dominant emotion independently, then share their initial response. If all agree, set \"is_disagreement_detected\" to false.

If any disagreement is detected, set \"is_disagreement_detected\" to true and begin a structured debate. During debate, each expert will:
- Write one step of their reasoning
- Share their reasoning with the group
- Revise their stance if needed

  Experts will repeat this process in **up to five rounds**, or until consensus is reached. If an expert determines their view is no longer valid or unreliable, they must exit the debate.

  After the debate ends, remaining experts must provide a final consensus answer.

 IMPORTANT: Emotion must be identified based on the following labels only: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no other text). 

  Once consensus is reached, provide the final output in the format below.
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
 “Is_disagreement_detected” : “True if initial disagreement detected between experts, False if no disagreement detected”,
 "debate": Provide object in format [

{
role: “Expert _”,
argument: “XXX"
},
{
role: “Expert _”,
argument: “XXX"
}, ...
]
  "final_justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions”,
}, 
“num_debate_rounds”: “ 1-5 (number of debate rounds)”
"""
)


SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT_DEBATE = (
"""
You are given a multimodal_description extracted from a video. Please provide a response in JSON format only, without any other text. Your task is to identify the dominant emotion being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

Three bimodal experts are analyzing the video, each focusing on a different modality:

  1. Expert in Text and Visual Interpretation (A) has access only to: \"whisper_transcript\", \"visual_expression_description\" and \"visual_objective_description\"
  2. Expert in Audio and Text Interpretation (B) has access only to: \"audio_description\" and  \"whisper_transcript\"
  3. Expert in Visual and Audio Interpretation (C) has access only to:  \"audio_description\", \"visual_expression_description\" and \"visual_objective_description\"

  All experts must first identify the dominant emotion independently, then share their initial response. If all agree, set \"Is_disagreement_detected\" to false.

  If any disagreement is detected, set \"Is_disagreement_detected\" to true and begin a structured debate. During debate, each expert will:
  - Write one step of their reasoning
  - Share their reasoning with the group
  - Revise their stance if needed

  Experts will repeat this process in **up to five rounds**, or until consensus is reached. If an expert determines their view is no longer valid or unreliable, they must exit the debate.

  After the debate ends, remaining experts must provide a final consensus answer.

 IMPORTANT: Emotion must be identified based on the following labels only: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no other text).

  Once consensus is reached, provide the final output in the format below.
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
 “Is_disagreement_detected” : “True if initial disagreement detected between experts, False if no disagreement detected”,
 "debate": Provide object in format [

{
role: “Expert _”,
argument: “XXX"
},
{
role: “Expert _”,
argument: “XXX"
}, ...
]
  "final_justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions”,
}, 
“num_debate_rounds”: “ 1-5 (number of debate rounds)”
"""
)


SYSTEM_PROMPT_CHAIN_OF_THOUGHT = (
   """
Let's think step by step.

You are given a multimodal_description extracted from a video.Please provide a response in JSON format only, without any other text. Your task is to identify the dominant emotion being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported. Please base your judgment on the input provided and respond according to the final_output format below. 

IMPORTANT NOTES: 
 IMPORTANT: Emotion must be identified based on the following labels only: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no other text). 
descriptions:
“whisper_transcript”;
“audio_description”;
“visual_expression_description”, “visual_objective_description” 

Based on the input provided, please respond according to the final_output format below:
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
  "justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions.", 
}
   """
)


SYSTEM_PROMPT_TREE_OF_THOUGHT = (
   """
 "You are given a multimodal_description extracted from a video. Please provide a response in JSON format only, without any other text. Imagine three different experts are answering a question.

   All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.
   The question is…

to identify the dominant emotion among  [neutral, angry, happy, sad, worried, surprise] being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

 IMPORTANT: Emotions must be identified based on the following labels only: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no other text).

All expert has access to the following variables in the multimodal descriptions:
  1. \"whisper_transcript\"
  2. \"audio_description\"
  3. \"visual_expression_description\" and \"visual_objective_description\"


Once there is consensus on the input provided, please respond according to the final_output format below:
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
  "final_justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions”,
}
"""
)


SYSTEM_PROMPT_TREE_OF_THOUGHT_UNIMODAL_3_EXPERT = (
   """
  You are given a multimodal_description extracted from a video. Imagine three different unimodal experts are analyzing the video, each focusing on a distinct modality:

   1. Expert in Text Interpretation (A) can only access to: \“whisper_transcript\”
   2. Expert in Audio Interpretation (B) can only access to: \“audio_description\”
   3. Expert in Visual Interpretation (C) can only access to: \“visual_expression_description\” and \“visual_objective_description\”

   All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.
   The question is...

to identify the dominant emotion being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

IMPORTANT NOTES: 
For emotion labels, Insert one of: [neutral, angry, happy, sad, worried, surprise] in lowercase only without any other text.

Once there is consensus on the input provided, please respond according to the final_output format below:
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
  "justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions."
}

   """
)


SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT = (
   """
    You are given a multimodal_description extracted from a video. Imagine three different experts are analyzing the video, each focusing on a different combination of modalities:

   1. Expert in Text and Visual Interpretation (A) can only access to \“whisper_transcript\” and \“visual_expression_description\” and \“visual_objective_description\”
   2. Expert in Text and Audio Interpretation (B) can only access to: \“whisper_transcript\” and  \“audio_description\”
   3. Expert in Visual and Audio Interpretation (C) can only access to : \“audio_description\”, \“visual_expression_description\”, and \“visual_objective_description\”
  
 All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.
   The question is...

to identify the dominant emotion being expressed. If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported. 

IMPORTANT NOTES: 
For emotion labels, Insert one of: [neutral, angry, happy, sad, worried, surprise] in lowercase only without any other text.

Once there is consensus on the input provided, please respond according to the final_output format below:
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
  "justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions."
}
   """
)
