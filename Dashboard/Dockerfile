# Build from the repo ROOT: docker build -f Dashboard/Dockerfile -t recsys-dashboard .
FROM python:3.12-slim
WORKDIR /app

# Only what dashboard.py actually reads at runtime -- keeps the image small
# and avoids needing the 80MB+ train.csv or raw dataset downloads.
COPY Dashboard/dashboard.py Dashboard/dashboard.py
COPY Dashboard/static Dashboard/static
COPY models/leaderboard.csv models/leaderboard.csv
COPY models/phase1_results.json models/phase1_results.json
COPY models/phase2_results.json models/phase2_results.json
COPY models/phase2b_results.json models/phase2b_results.json
COPY models/phase6a_results.json models/phase6a_results.json
COPY models/phase6b_results.json models/phase6b_results.json
COPY data/processed/item_meta.csv data/processed/item_meta.csv
COPY data/processed/mappings.json data/processed/mappings.json
COPY reports reports
COPY README.md README.md

RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn

WORKDIR /app/Dashboard
EXPOSE 7860
# Hugging Face Spaces expects port 7860; Render/Railway inject $PORT.
# This falls back to 7860 when $PORT isn't set, so the same image works on either.
CMD ["sh", "-c", "uvicorn dashboard:app --host 0.0.0.0 --port ${PORT:-7860}"]
