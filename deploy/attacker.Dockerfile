# Attacker swarm image. Scaled with `--scale attacker=N`; each replica gets its own
# IP on idsnet -> distributed (Mirai-style) traffic. hping3 + nmap preinstalled.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        hping3 nmap iproute2 bash \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /attacks
COPY attacks/ ./
RUN chmod +x ./*.sh
ENTRYPOINT ["./swarm_entrypoint.sh"]
