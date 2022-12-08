FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#CMD python build_structures.py && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3 --log-level debug
CMD uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3 --log-level debug
