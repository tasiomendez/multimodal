FROM python:3.6

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN apt update
RUN apt install -q python3-pip ffmpeg -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/tyiannak/pyAudioAnalysis.git /usr/src/pyAudioAnalysis
RUN pip3 install -e /usr/src/pyAudioAnalysis

CMD python3 main.py --file $CONFIG
