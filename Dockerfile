FROM continuumio/miniconda3

WORKDIR /opt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir pandas matplotlib

COPY setup.py .
COPY redol/ redol/
COPY main/main.py .
COPY util/ .
COPY VERSION .

RUN python setup.py install

CMD ["python", "-u", "main.py"]