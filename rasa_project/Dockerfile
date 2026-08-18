FROM rasa/rasa:3.6.21-full

WORKDIR /app

COPY --chown=1001:0 . /app

RUN rasa train --fixed-model-name campus-faq

# The base image entrypoint is `rasa`; reset it so the shell can expand Render's PORT.
ENTRYPOINT []
CMD ["sh", "-c", "rasa run --model models/campus-faq.tar.gz --enable-api --cors '*' --port ${PORT:-5005}"]
