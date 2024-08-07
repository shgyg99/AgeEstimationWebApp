# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7


FROM python:3.11.8

WORKDIR /app

COPY . .

RUN python -m pip install -r requirements.txt


# Run the application.
CMD ["streamlit", "run" ,"app.py"]
