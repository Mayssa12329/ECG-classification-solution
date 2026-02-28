FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_rpi.txt .
RUN pip install --no-cache-dir -r requirements_rpi.txt

# Copy project files
COPY . .

# Run benchmark
CMD ["python", "rasbery_pi/benchmark.py", "--n_runs", "100"]