FROM python:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python build_structures.py  && uvicorn main:app --host 0.0.0.0 --port 80 --workers 4
