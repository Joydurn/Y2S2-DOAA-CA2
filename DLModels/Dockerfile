FROM tensorflow/serving
COPY / /

ENV MODEL_NAME=cifar MODEL_BASE_PATH=/ MODEL_FILE_NAME=cifarModel
EXPOSE 8500
EXPOSE 8501
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server \
--rest_api_port=$PORT \
--model_name=${MODEL_NAME} \
--model_base_path=${MODEL_BASE_PATH}/{MODEL_FILE_NAME}\
--model_config_file_poll_wait_seconds=60 \
--model_config_file=${MODEL_BASE_PATH}/${MODEL_FILE_NAME}/models.config \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh