# 명령어 모음
## Project 만들기
npx degit reduxjs/redux-templates/packages/vite-template-redux pnp-front
## Docker Build
docker build -t pnp-front .
## Container
docker run --name front-container -it --rm -p 5173:5173 -v "$(pwd):/project" -v /project/node_modules pnp-front

docker run --name front-container -it --rm -p 5173:5173 -v "$(pwd):/project" -v /project/node_modules pnp-front sh


docker run --name front-container -it --rm -p 5173:5173 -v "$(pwd):/project" pnp-front sh
npm install
npm run dev

