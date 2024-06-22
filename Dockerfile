FROM python:3.11.9-slim

COPY . ./

RUN pip install -r ./package.txt

CMD ["python", "app.py"]