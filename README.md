# Random-Chess: An LLM benchmark

Random-chess is a simple yet revealing benchmark, that requires an LLM to reason from it's world knowledge of the chess rules and understanding of geometric relations, in order to calculate the valid moves for a chess position. This includes reasoning about kings being in check, pins, castling, and so on.

As the positions are randomly generated, memory based answers are not useful, and the training set cannot be picked up during training because it is generated on-the-fly.

## Key Features

- **Anti-Memorization Test**: Generates unique positions on-the-fly through random play.
- **Real Chess Understanding**: Tests move generation in positions the LLM can never have seen before, requiring actual reasoning.
- **Stochastic Benchmarking**: Prevents model gaming through unpredictable test sets (until someone embeds a chess engine in their API calls...), results are shown with confidence intervals.
- **Progress Display**: Streaming mode for real-time thinking observation, comprehensive logging for analysis.

## How It Works

1. **Position Generation**:
   - Creates 200 unique positions via random play with some piece weighting to ensure diverse positions.
   - Generates the legal moves through `python-chess`.

2. **LLM Evaluation**:
   - Presents FEN notation.
   - Requires UCI-format move lists with a specific prefix for the answer.
   - 1 point per position if the entire move list is correct.

## Installation

```bash
pip install python-chess openai

## Usage

# Minimum required arguments
python eval.py \
  --base-url "YOUR_API_URL" \
  --api-key "YOUR_API_KEY" \
  --model "MODEL_NAME"

# Full-featured example
python eval.py \
  --base-url "https://api.example.com/v1" \
  --api-key $LLM_API_KEY \
  --model "chess-specialist" \
  --results-log results.tsv \
  --conversation-log conversations.tsv \
  --concurrency 3 \
  --request-timeout 120 \
  --stream

# Environment Variables:
export LLM_API_BASE_URL="your_api_url"
export LLM_API_KEY="your_api_key"
export LLM_MODEL="model_name"
```

## Results



## License

MIT License

Copyright (c) 2025 Gian-Carlo Pascutto