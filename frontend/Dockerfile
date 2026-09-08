# Frontend image for local development via docker compose: build the static
# bundle, serve it with nginx. The public deployment doesn't use this — there
# the frontend is served by FastAPI from the same origin as the API, so there
# is no CORS boundary and no second container (see deploy/Dockerfile.space).

FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Empty VITE_API_BASE would mean same-origin, which is wrong here: in compose
# the API is a separate service on a different port, reached from the user's
# browser (not from inside the network), so it must be an absolute localhost
# URL.
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
