{%- if cookiecutter.use_bfio.lower()=="true" %}
FROM labshare/polus-bfio-util:1.3.6
{%- else %}
FROM python:3.7-alpine
{%- endif %}

COPY VERSION /
		
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 install -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]