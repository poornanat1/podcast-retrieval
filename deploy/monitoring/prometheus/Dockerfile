# Prometheus with the PodFind scrape configuration baked in. Build from the
# repository root. Listens on all addresses (Railway's private network is
# IPv6) and keeps 30 days on the /prometheus volume.
FROM prom/prometheus:v3.15.1
COPY deploy/monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
CMD ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus", \
     "--storage.tsdb.retention.time=30d", "--web.listen-address=[::]:9090", "--web.enable-lifecycle"]
