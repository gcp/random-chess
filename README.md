# Random-Chess: An LLM benchmark

Random-chess is a simple yet revealing benchmark, that requires an LLM to reason from its world knowledge of the chess rules and understanding of geometric relations between the squares on a grid, in order to calculate the valid moves for a chess position. This includes reasoning about kings being in check, pins, castling, and so on.

As the positions are randomly generated, memory based answers are not useful, and the training set cannot be picked up during training because it is generated on-the-fly.

## Key Features

- **Anti-Memorization Test**: Generates unique positions on-the-fly through random play.
- **Tests Real Understanding and Reasoning**: Tests move generation in positions the LLM can never have seen before, requiring actual reasoning *or* a deeper understanding of the board geometry *in* the model.
- **Stochastic Benchmarking**: Prevents model gaming through unpredictable test sets (until someone embeds a chess engine in their API calls...), results are shown with confidence intervals.
- **Progress Display**: Streaming mode for real-time thinking observation, comprehensive logging for analysis.

## How It Works

1. **Position Generation**:
   - Creates 200 unique positions via random play with some piece weighting to ensure diverse positions.
   - Generates the legal moves through `python-chess`.

2. **LLM Evaluation**:
   - Presents FEN notation.
   - Requires UCI-format move lists with a specific prefix for the answer.
   - F1 score for matching the legal move list, additionally computing the amount of exact answers.

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
  --model "gpt-4o" \
  --results-log gpt-4o.results \
  --conversation-log gpt-4o.conversation \
  --concurrency 1 \
  --request-timeout 120 \
  --stream

# Environment Variables
export LLM_API_BASE_URL="your_api_url"
export LLM_API_KEY="your_api_key"
export LLM_MODEL="model_name"

# Generating score report from logs
python score-runs.py --files results/*.results

```

## Results

| Rank | File                     | #Pos |    F1(%)         | Exact% |
|------|--------------------------|------|------------------|--------|
|    1 | o1-preview (2024-09-12)  | 200  |  90.3% ±  2.5%   |  16.0% |
|    2 | o1-mini (2024-09-12)     | 200  |  82.8% ±  2.6%   |   3.5% |
|    3 | deepseek-r1              | 200  |  72.6% ±  5.0%   |   6.5% |
|    4 | claude-3.5-sonnet        | 200  |  58.7% ±  1.9%   |   0.0% |
|    5 | deepseek-v3              | 200  |  45.5% ±  2.1%   |   0.0% |
|    6 | gpt-4o (2024-08-06)      | 200  |  37.1% ±  2.3%   |   0.0% |
|    7 | claude-3.5-haiku         | 200  |  27.8% ±  2.8%   |   0.0% |
|    8 | gemini-flash-2.0         | 200  |  11.3% ±  3.4%   |   0.0% |

## Future

Current LLMs can't reliably determine the legal moves in a chess position. This is a prerequisite for playing an actual game. If the state of the art advances enough that they are in the very high 90's, we can imagine extending the test, or eventually simply playing games.

Specific training on chess can obviously make [very high performance transformers](https://lczero.org/dev/wiki/best-nets-for-lc0/) dedicated to chess. This test is aimed at seeing how general LLM can deal with the required reasoning.

## License

MIT License

Copyright (c) 2025 Gian-Carlo Pascutto, Héðinn Steingrímsson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
