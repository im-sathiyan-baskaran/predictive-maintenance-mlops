FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train at build time so the image ships with a ready model.
# In a real deployment you'd pull artifacts/model.pkl from a registry/S3
# instead of retraining inside the image build.
RUN python train.py

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
