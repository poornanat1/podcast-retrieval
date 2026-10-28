# Builds any Go binary from cmd/; select with --build-arg PODFIND_CMD=<name>
# (on Railway, a PODFIND_CMD service variable). The API image embeds the
# web app from web/static.
FROM golang:1.26 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG PODFIND_CMD=api
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/app ./cmd/${PODFIND_CMD}

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
EXPOSE 8080
ENTRYPOINT ["/app"]
