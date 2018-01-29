# Pull base image
FROM ubuntu:latest

MAINTAINER Artem.Filatov@skoltech.ru

# Install Python.
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install latex
RUN apt-get update && apt-get install -y texlive

#RUN apt-get install python-lxml
RUN apt-get install -y git

RUN \
  pip3 install \
      numpy \
      pandas \
      matplotlib

RUN \
   pip3 install cvxpy
RUN pip3 install cvxopt


RUN \
  git clone     https://github.com/FilatovArtm/optMarkowitz

# Define working directory.
WORKDIR /optMarkowitz
VOLUME /optMarkowitz/results

RUN chmod +x run.sh

# Define default command.
CMD ./run.sh
