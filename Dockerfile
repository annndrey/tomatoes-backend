FROM python:3.7.3

ENV APPSETTINGS config_dev.py

COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

ADD . /app
ENV HOME /webapp
WORKDIR /app

EXPOSE 9988

CMD uwsgi --ini uwsgi.ini
