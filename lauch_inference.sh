
    # Fixed variable assignments and removed ${} in python command
    MODEL_PATH="YOU CAN  DOWNLAOD THE MODEL WIEGHTS FROM HUGGINGFACE AND PUT THE PATH HERE"
    PROMPT="what is in this image ?"
    IMAGE_FILE_PATH="D:\vs_code\Multimodal vision language model from scratch\test_cars[00_00_05][20240928-210552].png"
    MAX_TOKENS_TO_GENERATE=100
    TEMPERATURE=0.7
    TOP_P=0.9
    DO_SAMPLE="FALSE"
    ONLY_CPU="FALSE"

    python inference.py \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --image "$IMAGE_FILE_PATH" \
        --max-tokens "$MAX_TOKENS_TO_GENERATE" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --do-sample "$DO_SAMPLE" \
        --only-cpu "$ONLY_CPU"