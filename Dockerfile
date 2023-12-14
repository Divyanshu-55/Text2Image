FROM python:3.9

# Copy requirements.txt to the docker image and install packages
COPY requirements.txt /

RUN pip install -r requirements.txt

ADD https://drive.google.com/file/d/10E9NyRAd0_FgmbjVet5KnETaw8YSMgkA/view 800/unet/

# Set the WORKDIR to be the folder
COPY . /app

ENV PORT 8000

# Expose port 5000
EXPOSE $PORT

WORKDIR /app

# Use gunicorn as the entrypoint
CMD exec gunicorn Text2image-api:app --bind 0.0.0.0:$PORT  --workers 8 --threads 8 --timeout 240
