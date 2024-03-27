#!/bin/bash -e
# Ref: https://www.baeldung.com/ops/kafka-docker-setup

# Source helper
git_dir=$(git rev-parse --show-toplevel)
source ${git_dir}/helper.sh

# Compose docker to get the containers up
${DOCKER}-compose -f ${git_dir}/packaging/kafka-docker-compose.yml up -d

# Check if zookeeper and kafka are up
nc -zv localhost ${ZOOKEEPER_PORT}
nc -zv localhost ${KAFKA_PORT}

# Make sure all the topics are cleaned up
${DOCKER} exec tie-breaker-kafka bash -c "
    for topic in \$(kafka-topics --list --bootstrap-server localhost:${KAFKA_PORT}); do
        kafka-topics --delete --topic \${topic} --bootstrap-server localhost:${KAFKA_PORT};
    done"
