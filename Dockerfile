FROM python:3.13

EXPOSE 8080
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Dashboard_2/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
