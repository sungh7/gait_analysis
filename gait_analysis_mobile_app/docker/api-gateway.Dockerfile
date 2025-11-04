# ========================================
# API Gateway Dockerfile
# 고성능 GraphQL/REST API Gateway
# ========================================

# Build stage
FROM node:18-alpine AS builder

LABEL maintainer="Gait Analysis Team <team@gaitanalysis.com>"
LABEL description="High-performance API Gateway for Gait Analysis Pro"
LABEL version="1.0.0"

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

# Set working directory
WORKDIR /app

# Copy package files
COPY backend/api-gateway/package*.json ./
COPY backend/api-gateway/yarn.lock ./

# Install dependencies
RUN yarn install --frozen-lockfile --production=false

# Copy source code
COPY backend/api-gateway/ .

# Build application
RUN yarn build

# Remove dev dependencies
RUN yarn install --frozen-lockfile --production=true && \
    yarn cache clean

# ========================================
# Production stage
# ========================================
FROM node:18-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache \
    dumb-init \
    curl \
    ca-certificates

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Set working directory
WORKDIR /app

# Copy built application
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./package.json

# Copy configuration files
COPY --chown=nodejs:nodejs backend/api-gateway/config/ ./config/

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Start application
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/main.js"]

# ========================================
# Build arguments and environment variables
# ========================================
ARG BUILD_VERSION=unknown
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown

ENV NODE_ENV=production
ENV PORT=8080
ENV METRICS_PORT=9090
ENV LOG_LEVEL=info
ENV BUILD_VERSION=$BUILD_VERSION
ENV BUILD_DATE=$BUILD_DATE
ENV GIT_COMMIT=$GIT_COMMIT

# Labels for metadata
LABEL org.opencontainers.image.version=$BUILD_VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$GIT_COMMIT
LABEL org.opencontainers.image.source="https://github.com/gait-analysis/gait-analysis-pro"
LABEL org.opencontainers.image.title="Gait Analysis API Gateway"
LABEL org.opencontainers.image.description="GraphQL/REST API Gateway with authentication, rate limiting, and monitoring"