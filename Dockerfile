
FROM python:3.11-slim

# Create a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Install dependencies
COPY requirements-deploy.txt /app/requirements-deploy.txt

RUN pip install --no-cache-dir \
    -r /app/requirements-deploy.txt

# Copy application and trained model
COPY --chown=user:user app.py /app/app.py
COPY --chown=user:user models/best_model.joblib /app/models/best_model.joblib

# Use a non-root user
USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Launch Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]
