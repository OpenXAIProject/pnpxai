FROM node:20-alpine

WORKDIR /project
COPY package.json /project

RUN npm install

COPY . /project

ENV NODE_OPTIONS=--max-old-space-size=4096