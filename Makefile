# Deepiri Docker Compose Makefile
# Makes rebuilding clean and easy

.PHONY: rebuild clean build up down logs

# Clean rebuild - removes old images first (ONLY use when rebuilding needed)
rebuild:
	@echo "🧹 Cleaning old images..."
	docker compose -f docker-compose.dev.yml down --rmi local
	docker builder prune -af
	@echo "🔨 Rebuilding..."
	docker compose -f docker-compose.dev.yml build --no-cache
	@echo "✅ Rebuild complete!"

# Clean rebuild specific service (ONLY use when rebuilding needed)
rebuild-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "Usage: make rebuild-service SERVICE=pyagent"; \
		exit 1; \
	fi
	@echo "🧹 Cleaning old image for $(SERVICE)..."
	docker compose -f docker-compose.dev.yml rm -f -s -v $(SERVICE) 2>/dev/null || true
	docker rmi deepiri-dev-$(SERVICE):latest 2>/dev/null || true
	docker builder prune -af
	@echo "🔨 Rebuilding $(SERVICE)..."
	docker compose -f docker-compose.dev.yml build --no-cache $(SERVICE)
	@echo "✅ Rebuild complete!"

# Clean everything (removes containers, images, volumes, cache)
clean:
	@echo "🧹 Cleaning Docker resources..."
	docker compose -f docker-compose.dev.yml down --rmi local -v
	docker builder prune -af
	docker image prune -f
	@echo "✅ Clean complete!"

# Build (normal, with cache) - only rebuilds if needed
build:
	docker compose -f docker-compose.dev.yml build

# Up (normal start - uses existing images, NO rebuild)
up:
	docker compose -f docker-compose.dev.yml up -d

# Down
down:
	docker compose -f docker-compose.dev.yml down

# Logs
logs:
	docker compose -f docker-compose.dev.yml logs -f

# Show disk usage
df:
	docker system df

