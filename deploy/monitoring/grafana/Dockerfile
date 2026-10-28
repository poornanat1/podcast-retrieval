# Grafana provisioned with the Prometheus data source, the PodFind
# dashboard, and the alert rules that page the webhook. Build from the
# repository root. PROMETHEUS_URL and ALERT_WEBHOOK_URL come from the
# environment (ALERT_WEBHOOK_URL defaults to a sink so provisioning
# succeeds before a webhook is configured).
FROM grafana/grafana:13.3.0
ENV PROMETHEUS_URL=http://prometheus:9090 \
    ALERT_WEBHOOK_URL=http://127.0.0.1:9/unconfigured \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_USERS_ALLOW_SIGN_UP=false \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/podfind.json
COPY deploy/monitoring/grafana/provisioning /etc/grafana/provisioning
COPY deploy/monitoring/grafana/dashboards /var/lib/grafana/dashboards
