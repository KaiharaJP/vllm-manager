# Next.js Dockerfile

# ビルドステージ
FROM node:20-alpine AS builder

RUN apk add --no-cache ca-certificates libc6-compat

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG BACKEND_INTERNAL_URL
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_LITELLM_URL
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY} \
    BACKEND_INTERNAL_URL=${BACKEND_INTERNAL_URL} \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_PUBLIC_LITELLM_URL=${NEXT_PUBLIC_LITELLM_URL} \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

COPY package.json package-lock.json* ./

# npm registry への接続が不安定／プロキシ必須な環境向けのリトライ設定
RUN npm config set fetch-retries 10 \
    && npm config set fetch-retry-mintimeout 10000 \
    && npm config set fetch-retry-maxtimeout 180000 \
    && npm config set progress false

RUN npm ci

COPY . .
RUN npm run build

# 本番ステージ
FROM node:20-alpine AS runner

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
