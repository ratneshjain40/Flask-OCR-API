FROM ubuntu

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install python3-pip -y
RUN apt-get install python3-opencv -y

COPY . ./OCR_API
RUN apt-get install tesseract-ocr -y
WORKDIR /OCR_API
RUN pip install -r requirements.txt
RUN chmod +x gunicorn.sh
CMD [ "./gunicorn.sh" ]