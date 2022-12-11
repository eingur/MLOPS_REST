FROM python:3.10

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
ADD . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# COPY models /home/eingur/models
# COPY data /home/eingur/data

# COPY fit_and_save_model.py /home/eingur/

# COPY main_restx.py /home/eingur/
CMD python3 /app/main_restx.py