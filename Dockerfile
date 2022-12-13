FROM python:3.10


# Copy the current directory contents into the container at /app
WORKDIR /app

COPY . .
# ADD . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

CMD python3 /app/main.py