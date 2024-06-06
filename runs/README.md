# Results

## Full results

| Model | Google Search <br> Grounding | Percentage Correct <br> (Min, Avg, Max) | Execution Seconds <br> (Min, Avg, Max) |
|:------|:------------------------|:-----------------------------------|:----------------------------------|
| gemini-1.0-pro   | False        | 88, 91, 96                         | 13.69, 14.35, 15.74               |
| gemini-1.0-pro   | True         | 76, 84, 92                         | 16.27, 16.61, 17.4                |
| gemini-1.0-ultra | False        | 92, 94, 96                         | 43.92, 44.62, 45.74               |
| gemini-1.5-flash | False        | 60, 82, 96                         | 13.97, 16.77, 18.09               |
| gemini-1.5-flash | True         | 68, 83, 92                         | 16.81, 19.19, 23.76               |
| gemini-1.5-pro   | False        | 88, 93, 96                         | 37.94, 39.98, 42.48               |
| gemini-1.5-pro   | True         | 84, 88, 92                         | 38.71, 42.08, 48                  |

## Average results without grounding

| Model | Percentage Correct <br> (Avg) | Execution Seconds <br> (Avg) |
|:------|:------------------------|:------------------------------|
| gemini-1.0-pro   | 91           | 14.35                         |
| gemini-1.0-ultra | 94           | 44.62                         |
| gemini-1.5-flash | 82           | 16.77                         |
| gemini-1.5-pro   | 93           | 39.98                         |