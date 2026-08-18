FROM rasa/rasa:3.6.21-full

WORKDIR /app

COPY . /app

RUN rasa train --fixed-model-name campus-faq

CMD ["sh", "-c", "rasa run --model models/campus-faq.tar.gz --enable-api --cors '*' --port ${PORT:-5005}"]
