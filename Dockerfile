FROM continuumio/miniconda3

WORKDIR /opt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir pandas matplotlib

COPY setup.py .
COPY redol/ redol
COPY main/ main
COPY util/ util
COPY data/ data

RUN python setup.py install