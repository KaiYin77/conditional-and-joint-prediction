FROM pytorch/pytorch:latest
MAINTAINER kevin880208.eed07@nctu.edu.tw

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt
RUN pip install tensorboard
