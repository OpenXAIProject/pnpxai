up-w-logs:
	docker-compose up
up:
	docker-compose up -d
build:
	docker-compose up --build -d
down:
	docker-compose down
logs:
	docker-compose logs pnpxai
bash:
	docker-compose exec pnpxai bash