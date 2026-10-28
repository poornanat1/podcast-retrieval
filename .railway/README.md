# Railway configuration

`railway.ts` describes the PodFind project on Railway: services, volumes,
buckets, and environment variables. How to deploy, load the catalog, roll
back, and monitor is in [docs/11-deployment.md](../docs/11-deployment.md).

```sh
cd .railway && npm install && cd ..   # the railway/iac SDK, for the CLI's type check
railway login && railway link         # once
railway config plan                   # preview
railway config apply                  # create or update
```
