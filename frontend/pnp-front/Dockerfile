FROM node:20-alpine

WORKDIR /project
COPY package.json /project

RUN npm install

COPY . /project

CMD [ "npm", "run", "dev" ]
