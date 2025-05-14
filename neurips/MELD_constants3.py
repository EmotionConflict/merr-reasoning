SYSTEM_PROMPT = (
   """
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported. Please base your judgment on the input provided and respond according to the final_output format below. 

[IMPORTANT] 
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral]
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Output Format]
Based on the input provided, please provide *final_output* in the following format: 
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
[Task Overview]
You are given a multimodal_description extracted from a video from a TV show. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid), force to choose from the given list of emotions.
- If there is not distinctly dominant emotion than label as "neutral"
- Each expert should try not to be biased:
   - Contextual Bias: Assuming the emotional tone based on the setting (e.g., romantic or humorous scenes assumed to indicate joy).
   - Figurative Language Bias: Taking metaphorical expressions at face value and misclassifying emotions (e.g., 'shrieking' = fear).
   - Visual Overweighting Bias: Over-relying on facial expressions without cross-checking audio or dialogue content.
   - Listener-Speaker Attribution Bias: Misattributing the emotional expressions of a listener to the speaking individual.
   - Intensity Misinterpretation Bias: Interpreting strong vocal or facial intensity as anger or sadness when it may indicate confidence or emphasis.
   - Neglect of Neutrality Bias: Hesitating to classify an emotion as neutral, resulting in overclassification as sadness or mild negative emotion.
   - Sarcasm Misinterpretation Bias: Assuming sarcastic tone always implies anger or annoyance, ignoring playful or humorous intent.
   - Confirmation Bias: Favoring cues that support a preconceived emotional hypothesis while downplaying contradictory cues.
   - Single-Modality Reliance Bias: Drawing conclusions from only one modality (e.g., audio, visual, or text) instead of integrating all three.
   - Temporal Overgeneralization Bias: Treating brief, transient expressions (like surprise) as dominant emotional states rather than fleeting reactions.


[Emotion Classification Methodology]
Three critical unimodal experts are analyzing the video, each focusing on a different modality:
  1. Expert in Text Interpretation (A) has access only to: \"whisper_transcript\"
  2. Expert in Audio Interpretation (B) has access only to: \"audio_description\"
  3. Expert in Visual Interpretation (C) has access only to: \"visual_expression_description\" and \"visual_objective_description\"

   All experts must first identify the dominant emotion independently (only allowed to choose from [disgust, surprise, anger, joy, fear, sadness, neutral]), then share their initial response, along with evidence and a confidence score. If all agree, set \"is_disagreement_detected\" to false.
   If any disagreement is detected, set \"is_disagreement_detected\" to true and begin a structured debate where they can only discuss about [disgust, surprise, anger, joy, fear, sadness, neutral]. 
   During debate, each expert will:
   - Write one step of their reasoning based on evidence and report confidence score (1–100 with 100 being highest confidence) of their reasoning 
   - Share their reasoning with the group in random order 
   - Revise their stance if needed.
   - This is one round. Continue next rounds if needed.
      
     Experts will repeat this process in **at least three rounds**, and until consensus is reached. 
     If an expert determines their view is no longer valid or unreliable, they must state they are not confident and exit the debate.
     After the debate ends, remaining experts must provide a final consensus answer.
   
     Once consensus is reached, provide the final output in the format below.
  
[Output Format]
{
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
 “Is_disagreement_detected” : “True if initial disagreement detected between experts, False if no disagreement detected”,
“num_debate_rounds”: “ 1-10 (number of debate rounds, round means A & B & C took turns)”, 
"first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)"
"""
)



SYSTEM_PROMPT_TREE_OF_THOUGHT_BIMODAL_3_EXPERT_DEBATE = (
"""
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Emotion Classification Methodology]
Three critical bimodal experts are analyzing the video, each focusing on a different modality:

  1. Expert in Text and Visual Interpretation (A) has access only to: \"whisper_transcript\", \"visual_expression_description\" and \"visual_objective_description\"
  2. Expert in Audio and Text Interpretation (B) has access only to: \"audio_description\" and  \"whisper_transcript\"
  3. Expert in Visual and Audio Interpretation (C) has access only to:  \"audio_description\", \"visual_expression_description\" and \"visual_objective_description\"

  All experts must first identify the dominant emotion independently, then share their initial response, along with evidence and a confidence score. If all agree, set \"is_disagreement_detected\" to false.
  If any disagreement is detected, set \"is_disagreement_detected\" to true and begin a structured debate. During debate, each expert will:
   - Write one step of their reasoning based on evidence 
   - Share their reasoning with the group in random order
   - Revise their stance if needed

  Experts will repeat this process in **at most five rounds**, and until consensus is reached. If an expert determines their view is no longer valid or unreliable, they must state they are not confident and exit the debate.
  After the debate ends, remaining experts must provide a final consensus answer.
  Once consensus is reached, provide the final output in the format below.

[Output Format]
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
“num_debate_rounds”: “ 1-10 (number of debate rounds)”
"""
)


SYSTEM_PROMPT_CHAIN_OF_THOUGHT = (
   """
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Emotion Classification Methodology]
Let's think step by step.
You have access to the following variables in the multimodal descriptions:
  1. \"whisper_transcript\"
  2. \"audio_description\"
  3. \"visual_expression_description\" and \"visual_objective_description\"
Based on the input provided, please respond according to the final_output format below.

[Output Format]
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
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Emotion Classification Methodology]
All expert has access to the following variables in the multimodal descriptions:
  1. \"whisper_transcript\"
  2. \"audio_description\"
  3. \"visual_expression_description\" and \"visual_objective_description\"
  
All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.
  
Once there is consensus on the input provided, please respond according to the final_output format below.

[Output Format]
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
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Emotion Classification Methodology]
   1. Expert in Text Interpretation (A) can only access to: \“whisper_transcript\”
   2. Expert in Audio Interpretation (B) can only access to: \“audio_description\”
   3. Expert in Visual Interpretation (C) can only access to: \“visual_expression_description\” and \“visual_objective_description\”

   All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.
   
   Once there is consensus on the input provided, please respond according to the final_output format below.

[Output Format]
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
[Task Overview]
You are given a multimodal_description extracted from a video. Please provide a *final output* in JSON format only, without any other text. Your task is to identify the *dominant emotion* present in the video using the multimodal_description, where *dominant emotion* is one of the following: [disgust, surprise, anger, joy, fear, sadness, neutral]. 
If multiple emotions are present, select the most dominant one, and a second emotion only if it is clearly supported.

[IMPORTANT]
- Only choose emotions from the following exact list: [disgust, surprise, anger, joy, fear, sadness, neutral] (in lowercase, no extra text)
- Do not invent new emotion names (e.g., “happy”, “worried”, “anxious” are NOT valid)

[Emotion Classification Methodology]
   1. Expert in Text and Visual Interpretation (A) can only access to \“whisper_transcript\” and \“visual_expression_description\” and \“visual_objective_description\”
   2. Expert in Text and Audio Interpretation (B) can only access to: \“whisper_transcript\” and  \“audio_description\”
   3. Expert in Visual and Audio Interpretation (C) can only access to : \“audio_description\”, \“visual_expression_description\”, and \“visual_objective_description\”
  
   All experts will write down 1 step of their thinking,
   then share it with the group.
   Then all experts will go on to the next step, etc.
   If any expert realises they're wrong at any point then they leave.
   The remaining experts must reach a consensus.

   Once there is consensus on the input provided, please respond according to the final_output format below.
   
[Output Format]
{
  "first_emotion": "the most dominant emotion (please answer in lowercase one word)",
  "first_emotion_confidence_score": "1–100 (with 100 being highest confidence)",
  "second_emotion": "the second most evident emotion, if any (please answer in lowercase one word)",
  "second_emotion_confidence_score": "1–100 (leave blank if no second_emotion)",
  "justification": "Provide a brief reasoning (1–2 sentences) for your chosen emotions."
}
   """
)
