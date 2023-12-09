# frontend/pnp-front/Dockerfile
FROM node:20-alpine

WORKDIR /project

COPY package.json /project
COPY package-lock.json /project
COPY yarn.lock /project
COPY . /project

RUN npm install
RUN yarn install

ENV NODE_OPTIONS=--max-old-space-size=4096

CMD ["npm", "run", "dev"]