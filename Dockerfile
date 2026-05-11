# 1. Start with official Python 3.10 image (like buying a basic kitchen)
FROM python:3.10-slim

# 2. Set working directory (create a folder /app inside container)
WORKDIR /app

# 3. Copy only requirements first (caching optimization)
COPY requirements.txt .

# 4. Install Python packages (like buying groceries)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL our code into the container
COPY . .

# 6. Tell Docker we'll use port 8000 (like reserving a door for visitors)
EXPOSE 8000

# 7. Command to run when container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]