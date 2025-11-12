FROM node:18-alpine

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apk add --no-cache --virtual .build-deps curl dumb-init && \
    rm -rf /var/cache/apk/* && \
    apk del .build-deps 2>/dev/null || true

COPY package*.json ./

# Install dependencies and clean up in one layer
RUN npm install --omit=dev --legacy-peer-deps && \
    npm cache clean --force && \
    rm -rf /tmp/* /var/tmp/*

# Ensure utils directory exists (will be available via volume mount in dev)
RUN mkdir -p ./utils
COPY . .

# Create non-root user and set up directories in one layer
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    mkdir -p logs && chown -R nodejs:nodejs /app

USER nodejs

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]

